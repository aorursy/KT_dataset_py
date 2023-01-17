!git clone https://github.com/ultralytics/google-images-download
#!cd './google-images-download/'
!pip install -U -r ./google-images-download/requirements.txt
!wget https://chromedriver.storage.googleapis.com/2.41/chromedriver_linux64.zip -O ./chromedriver_linux64.zip
!unzip ./chromedriver_linux64.zip chromedriver
#!sudo ln -s chromedriver /usr/local/bin/
#!ls
!wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
!echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | sudo tee /etc/apt/sources.list.d/google-chrome.list
!sudo apt-get update
!sudo apt-get install google-chrome-stable
!sudo python ./google-images-download/bing_scraper.py --search 'honeybees on flowers' --limit 10 --download --chromedriver ./chromedriver
!zip -r images.zip ./images #поджать в архив папку, по желанию