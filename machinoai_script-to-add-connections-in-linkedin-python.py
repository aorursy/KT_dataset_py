from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import time

from selenium import webdriver



from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.by import By





driver=webdriver.Chrome('Enter your chromium driver path')

driver.get("https://www.linkedin.com/uas/login")



time.sleep(10)



username=driver.find_element_by_id("session_key-login")

password=driver.find_element_by_id("session_password-login")



username.send_keys("MyEmailId")

password.send_keys("myPassword")



login_attempt=driver.find_element_by_xpath("//*[@type='submit']")

login_attempt.submit()



driver.get("https://www.linkedin.com/mynetwork/")

#driver.find_element_by_css_selector(.button.c_button).click

driver.find_element_by_id("Enter the id or regex to match eg. 'emmber%'").click






















