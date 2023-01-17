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
browser.get("https://www.ifood.com.br/delivery/santarem-pa")
#browser.find_element_by_id("search_button_homepage").click()
#print(browser.current_url)
# WE CAN EVEN TAKE A "SCREENSHOT"!
browser.save_screenshot("screenshot.png")

!ls -l .
# LET'S LOOK AT IT!
from IPython.display import Image
Image("screenshot.png", width=800, height=500)
# CLOSE FIREFOX BROWSER
browser.quit()

!ps -A