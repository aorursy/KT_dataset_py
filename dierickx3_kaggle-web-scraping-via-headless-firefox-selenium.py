# SEE WHAT'S UNDER THE HOOD HERE
!cat /etc/os-release
# WHERE ARE WE RIGHT NOW?
!ls -l .
# BUT, WHERE, REALLY, ARE WE RIGHT NOW? (p.s. IT LOOKS LIKE WE'RE INSIDE /kaggle/working FOLDER, INSIDE AN ISOLATE DOCKER CONTAINER/IMAGE)
!echo "ls -l /kaggle"
!ls -l /kaggle

!echo "\nls -l /kaggle/working"
!ls -l /kaggle/working
# LOOK AT INPUT FOLDER, WE SHOULD SEE "firefox-63.0.3" FOLDER ALREADY THERE
!ls -l "../input"
# WE WILL MAKE NEW SUBFOLDER IN WORKING FOLDER (WHICH ISN'T READ-ONLY)
!mkdir "../working/firefox"
!ls -l "../working"
# COPY OVER FIREFOX FOLDER INTO NEW SUBFOLDER JUST CREATED
!cp -a "../input/firefox-63.0.3/firefox/." "../working/firefox"
!ls -l "../working/firefox"
# ADD READ/WRITE/EXECUTE CAPABILITES
!chmod -R 777 "../working/firefox"
!ls -l "../working/firefox"
# INSTALL PYTHON MODULE FOR AUTOMATIC HANDLING OF DOWNLOADING AND INSTALLING THE GeckoDriver WEB DRIVER WE NEED
!pip install webdriverdownloader
# INSTALL LATEST VERSION OF THE WEB DRIVER
from webdriverdownloader import GeckoDriverDownloader
gdd = GeckoDriverDownloader()
gdd.download_and_install("v0.23.0")
# INSTALL SELENIUM MODULE FOR AUTOMATING THINGS
!pip install selenium
# LAUNCHING FIREFOX, EVEN INVISIBLY, HAS SOME DEPENDENCIES ON SOME SCREEN-BASED LIBARIES
!apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb
# SETUP A VIRTUAL "SCREEN" FOR FIREFOX TO USe
!export DISPLAY=:99
# PYTHON MODULES TO USE
from selenium import webdriver as selenium_webdriver
from selenium.webdriver.firefox.options import Options as selenium_options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities
# FIRE UP A HEADLESS BROWSER SESSION WITH A "SCREEN SIZE" OF 1920x1080

browser_options = selenium_options()
browser_options.add_argument("--headless")
browser_options.add_argument("--window-size=1920,1080")

capabilities_argument = selenium_DesiredCapabilities().FIREFOX
capabilities_argument["marionette"] = True

browser = selenium_webdriver.Firefox(
    options=browser_options,
    firefox_binary="../working/firefox/firefox",
    capabilities=capabilities_argument
)
# SHOW LIST OF RUNNING PROCESSES; SHOULD SEE firefox AND geckodriver
!ps -A
# PERFORM A WEB SEARCH (SEE HOW WE CAN EVEN ARBITRARILY CHANGE BROWSER WINDOW SIZE ON-THE-FLY "MOSTLY" AS WE PLEASE, IF <= BROWSER_OPTION ABOVE)
browser.set_window_size(1366, 768)
browser.get("https://duckduckgo.com/")
browser.find_element_by_id('search_form_input_homepage').send_keys("Kaggle Rocks!")
browser.find_element_by_id("search_button_homepage").click()
print(browser.current_url)
# WE CAN EVEN TAKE A "SCREENSHOT"!
browser.save_screenshot("screenshot.png")

!ls -l .
# LET'S LOOK AT IT!
from IPython.display import Image
Image("screenshot.png", width=800, height=500)
# CLOSE FIREFOX BROWSER
browser.quit()

!ps -A
# MORE MODULES TO IMPORT
import pandas as pd
import seaborn as sns

from selenium.webdriver.common.by  import By as selenium_By
from selenium.webdriver.support.ui import Select as selenium_Select
from selenium.webdriver.support.ui import WebDriverWait as selenium_WebDriverWait
from selenium.webdriver.support    import expected_conditions as selenium_ec
# FIRE UP A HEADLESS BROWSER SESSION WITH A "SCREEN SIZE" OF 1920x1080

browser_options = selenium_options()
browser_options.add_argument("--headless")
browser_options.add_argument("--window-size=1920,1080")

capabilities_argument = selenium_DesiredCapabilities().FIREFOX
capabilities_argument["marionette"] = True

browser = selenium_webdriver.Firefox(
    options=browser_options,
    firefox_binary="../working/firefox/firefox",
    capabilities=capabilities_argument
)
# NAVIGATE TO NBA.COM TEAM STATS AND TAKE A PICTURE TO "PROVE" TO OURSELVES WHERE WE'RE AT
browser.get("https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1")
print(browser.current_url)
browser.save_screenshot("screenshot.png")
Image("screenshot.png", width=800, height=500)
# INSTEAD OF "Per Game" TEAM STATS, LET'S LOOK AT "Per 100 Possessions"
sel_obj = selenium_Select(browser.find_element_by_name("PerMode"))
sel_obj.select_by_visible_text("Per 100 Poss")
                          
# WE NEED TO WAIT FOR DYNAMIC CONTENT TO REFRESH; WE WILL GIVE IT UP TO 10 SECONDS
wait = selenium_WebDriverWait(browser, 10)
wait.until(selenium_ec.visibility_of_element_located((selenium_By.XPATH, '//div[@class="nba-stat-table__overflow"]')))

# WHILE AT IT, LET'S SORT BY PTS IN DESCENDING ORDER, JUST FOR FUN
browser.find_element_by_xpath('//th[@data-field="PTS"]').click()

# SEE WHERE WE ARE NOW
browser.save_screenshot("screenshot.png")
Image("screenshot.png", width=800, height=500)
# OK, LET'S GRAB THE HTML OF THIS TABLE NOW AND TAKE A PEAK AT FIRST 1500 CHARACTERS TO SEE IF IT SEEMS RIGHT
html = browser.find_element_by_xpath('//div[@class="nba-stat-table__overflow"]').get_attribute("outerHTML")
html[0:1500]
# LET'S LOAD INTO PANDAS AS SEE WHAT WE REALLY HAVE
df = pd.read_html(html)[0]
df.head()
# LET'S DO SOME CLEANUP HERE (WE ONLY CARE ABOUT COLUMNS 2-29)
df = df.iloc[:, 1:28]
df.head()
# DO SOME BASIC STATS
df.describe()
# WHAT'S "MOST IMPORTANT" IN TERMS OF TEAM OFFENSIVE SCORING PER 100 POSSESSIONS AMONG SOME MAIN STAT CATEGORIES?
# NOTE: FG% APPEARS MOST IMPORTANT - JUST LIKE "Basketball on Paper", BY DEAN OLIVER SUGGESTS!
tmp = df[["PTS","FG%","3P%","FT%","OREB","AST","TOV"]]
g = sns.pairplot(tmp, kind="reg")
# WE CAN CLOSE FIREFOX NOW (REALLY COULD HAVE AFTER WE SNAGGED THE HTML WE NEEDED)
browser.quit()

!ps -A
