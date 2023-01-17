from selenium import webdriver

from selenium.webdriver.common.keys import Keys

import time

from selenium.common.exceptions import NoSuchElementException

from selenium.common.exceptions import StaleElementReferenceException

from selenium.common.exceptions import TimeoutException

from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.by import By

from selenium.webdriver.support.select import Select
driver=webdriver.Chrome('f://chromedriver.exe')

#set the location where the chrome driver is located.
driver.get('https://www.instagram.com/')

driver.maximize_window()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.LINK_TEXT, "Log in")))
driver.find_element_by_link_text('Log in').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "username")))

time.sleep(2)
SAMPLE_USERNAME='usename'

SAMPLE_PASSWORD='password'

#change the usename and password to appropriate ones. Just change them here and u'll not need to change it anywhere else.

username_box=driver.find_element_by_name('username')

username_box.send_keys(SAMPLE_USERNAME)

password_box=driver.find_element_by_name('password')

password_box.send_keys(SAMPLE_PASSWORD)

driver.find_element_by_xpath('//button[contains(@class, "L3NKy")]/div').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "HoLwm")))
driver.find_element_by_class_name('HoLwm').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "TqC_a")))
search_box=driver.find_element_by_class_name('TqC_a').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "XTCLo")))
search_box=driver.find_element_by_class_name('XTCLo')

search_box.send_keys('food')

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[@class="yCE8d  "]/div/div[2]/div/span')))
for i in driver.find_elements_by_xpath('//a[@class="yCE8d  "]/div/div[2]/div/span'):

    print(i.get_attribute('innerHTML'))

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "coreSpriteSearchClear")))
driver.find_element_by_class_name('coreSpriteSearchClear').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "TqC_a")))
driver.find_element_by_class_name('TqC_a').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "XTCLo")))
search_box=driver.find_element_by_class_name('XTCLo')

search_box.send_keys('So Delhi')

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "yCE8d")))
driver.find_element_by_class_name('yCE8d').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "_6VtSN")))
if driver.find_element_by_class_name('_6VtSN').get_attribute('innerHTML')=='Following':

    print('You are already following the page!')

else:

    driver.find_element_by_xpath('//button[contains(@class, "_6VtSN")]').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "_6VtSN")))

time.sleep(3)
if driver.find_element_by_class_name('_6VtSN').get_attribute('innerHTML')=='Following':

    driver.find_element_by_class_name('_6VtSN').click()

    driver.find_element_by_xpath('//button[contains(@class, "-Cab_")]').click()

else:

    print('You have already unfollowed the page')
driver.find_element_by_class_name('TqC_a').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "XTCLo")))
search_box=driver.find_element_by_class_name('XTCLo')

search_box.send_keys('dilsefoodie')

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "yCE8d")))
driver.find_element_by_class_name('yCE8d').click()

time.sleep(3)
driver.execute_script('window.scrollBy(0, 3000);')

time.sleep(2.5)

driver.execute_script('window.scrollBy(0, 3000);')

time.sleep(2.5)

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "Nnq7C")]/div/a/div[1]')))
#liking 30 posts

count=0

for i in driver.find_elements_by_xpath('//div[contains(@class, "Nnq7C")]/div/a/div[1]'):

    i.click()

    trier=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,

                                                                    "//button[contains(@class, 'dCJp8')]")))

    

    try:

        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//section[contains(@class, "ltpMr")]')))

        time.sleep(0.8)

        driver.find_element_by_xpath('//button[contains(@class, "dCJp8")]/span[contains(@class, "glyphsSpriteHeart__outline__24__grey_9")]').click()

    except NoSuchElementException:

        print('Already Liked!')

    driver.back()

    count+=1

    if count==30:

        print('Liked All(30)!')

        break

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "Nnq7C")]/div/a/div[1]')))

time.sleep(2)

driver.execute_script('window.scrollTo(0, 0)')
#unliking 30 posts

count=0

for i in driver.find_elements_by_xpath('//div[contains(@class, "Nnq7C")]/div/a/div[1]'):

    i.click()

    trier=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,

                                                                    "//button[contains(@class, 'dCJp8')]")))

    

    try:

        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//section[contains(@class, "ltpMr")]')))

        driver.find_element_by_xpath('//button[contains(@class, "dCJp8")]/span[contains(@class, "glyphsSpriteHeart__filled__24__red_5")]').click()

    except NoSuchElementException:

        print('Already UnLiked!')

    driver.back()

    count+=1

    time.sleep(0.3)

    if count==30:

        print('UnLiked All(30)!')

        break

time.sleep(2)
driver.find_element_by_class_name('TqC_a').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "XTCLo")))
search_box=driver.find_element_by_class_name('XTCLo')

search_box.send_keys('So Delhi')

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "yCE8d")))
driver.find_element_by_class_name('yCE8d').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "_6VtSN")))

time.sleep(3)
try:

    driver.execute_script('window.scrollTo(0, 0)')

    driver.find_element_by_xpath('//a[@class="-nal3 "]/span[@class="g47SY "]').click()

except NoSuchElementException:

    driver.find_element_by_xpath('//a[@class=" _81NM2"]/span[contains(@class, "g47SY")]').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[contains(@class, "notranslate")]')))
while True:

    try:

        sodelhi=[]

        count=0

        while True:

            elements=driver.find_elements_by_xpath('//a[contains(@class, "notranslate")]')

            if len(elements)<500:

                waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//ul[contains(@class, "jSC57")]')))

                driver.execute_script('arguments[0].scrollIntoView(0, 100);', driver.find_element_by_xpath('//ul[contains(@class, "jSC57")]'))

                time.sleep(0.8)

            i=elements[count]

            sodelhi.append(i.get_attribute('innerHTML'))

            if len(sodelhi)>=500:

                break

            count+=1

        break

    except StaleElementReferenceException:

        continue

sodelhi

driver.find_element_by_class_name('glyphsSpriteX__outline__24__grey_9').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "TqC_a")))
driver.find_element_by_class_name('TqC_a').click()



search_box=driver.find_element_by_class_name('XTCLo')

search_box.send_keys('foodtalkindia')

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "yCE8d")))
driver.find_element_by_class_name('yCE8d').click()

waiter=WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//a[@class="-nal3 "]/span[@class="g47SY "]')))

time.sleep(2)
driver.find_element_by_xpath('//a[@class="-nal3 "]/span[@class="g47SY "]').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[contains(@class, "notranslate")]')))
while True:

    try:

        foodtalkindia=[]

        count=0

        while True:

            elements=driver.find_elements_by_xpath('//a[contains(@class, "notranslate")]')

            if len(elements)<500:

                waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//ul[contains(@class, "jSC57")]')))

                driver.execute_script('arguments[0].scrollIntoView(0, 100);', driver.find_element_by_xpath('//ul[contains(@class, "jSC57")]'))

                time.sleep(0.8)

            i=elements[count]

            foodtalkindia.append(i.get_attribute('innerHTML'))

            if len(foodtalkindia)>=500:

                break

            count+=1

        break

    except StaleElementReferenceException:

        continue

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "glyphsSpriteX__outline__24__grey_9")))

foodtalkindia
driver.find_element_by_class_name('glyphsSpriteX__outline__24__grey_9').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "glyphsSpriteUser__outline__24__grey_9")))

time.sleep(3)
driver.find_element_by_class_name('glyphsSpriteUser__outline__24__grey_9').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[@class="-nal3 "]/span')))

time.sleep(3)
total_followers=int(driver.find_element_by_xpath('//a[@class="-nal3 "]/span[@class="g47SY "]').get_attribute('innerHTML'))

driver.find_element_by_xpath('//a[@class="-nal3 "]/span[@class="g47SY "]').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[contains(@class, "notranslate")]')))
while True:

    try:

        myfollowers=[]

        count=0

        while True:

            elements=driver.find_elements_by_xpath('//a[contains(@class, "notranslate")]')

            if len(elements)<total_followers:

                waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//ul[contains(@class, "jSC57")]')))

                driver.execute_script('arguments[0].scrollIntoView(0, 100);', driver.find_element_by_xpath('//ul[contains(@class, "jSC57")]'))

                time.sleep(0.8)

            i=elements[count]

            myfollowers.append(i.get_attribute('innerHTML'))

            if len(myfollowers)==total_followers+1:

                break

            count+=1

        break

    except StaleElementReferenceException:

        continue

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'glyphsSpriteX__outline__24__grey_9')))
driver.find_element_by_class_name('glyphsSpriteX__outline__24__grey_9').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[@class="-nal3 "]/span')))
total_following=int(driver.find_elements_by_xpath('//a[@class="-nal3 "]/span')[1].get_attribute('innerHTML'))

driver.find_elements_by_xpath('//a[@class="-nal3 "]/span[@class="g47SY "]')[1].click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[contains(@class, "notranslate")]')))
while True:

    try:

        following=[]

        count=0

        while True:

            elements=driver.find_elements_by_xpath('//a[contains(@class, "notranslate")]')

            if len(elements)<total_following:

                waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//ul[contains(@class, "jSC57")]')))

                driver.execute_script('arguments[0].scrollIntoView(0, 100);', driver.find_element_by_xpath('//ul[contains(@class, "jSC57")]'))

                time.sleep(0.8)

            i=elements[count]

            following.append(i.get_attribute('innerHTML'))

            if len(following)==total_following+2:

                break

            count+=1

        break

    except StaleElementReferenceException:

        continue
following=set(following)

myfollowers=set(myfollowers)

foodtalkindia=set(foodtalkindia)

followers_of_foodtalkindia_that_i_am_following= following.intersection(foodtalkindia)

print('Given below is a set of all the followers of “foodtalkindia” that I am following but those who don’t follow me.')

if len(followers_of_foodtalkindia_that_i_am_following-myfollowers)==0:

    print("--->No such followers")

for i in followers_of_foodtalkindia_that_i_am_following-myfollowers:

    print(i)

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "glyphsSpriteX__outline__24__grey_9")))
driver.find_element_by_class_name('glyphsSpriteX__outline__24__grey_9').click()

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "TqC_a")))
driver.find_element_by_class_name('TqC_a').click()



search_box=driver.find_element_by_class_name('XTCLo')

search_box.send_keys('coding.ninjas')

waiter=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "yCE8d")))
driver.find_element_by_class_name('yCE8d').click()

time.sleep(3)
try:

    if int(driver.find_element_by_xpath('//div[contains(@class, "h5uC0")]/canvas').get_attribute('height'))==168:

        print('You have not seen the story yet! The story will be shown to you now. check out the driver window')

        driver.find_element_by_xpath('//div[contains(@class, "h5uC0")]').click()

    elif int(driver.find_element_by_xpath('//div[contains(@class, "h5uC0")]/canvas').get_attribute('height'))==166:

        print('You have already seen the story!')

except NoSuchElementException:

    print('The user has no story!')