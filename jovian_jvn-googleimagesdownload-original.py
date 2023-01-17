import jovian
!pip install google_images_download
!googleimagesdownload 
!googleimagesdownload  --keywords "Polar Bear" --limit 150
!googleimagesdownload  --keywords "Polar Bear" --limit 150 --chromedriver "/usr/bin/chromedriver"
jovian.commit()