import os

os.listdir('/kaggle/input')
# !wget --header="Host: com-mendeley-internal.s3.eu-west-1.amazonaws.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) coc_coc_browser/85.0.130 Chrome/79.0.3945.130 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7" --header="Referer: https://data.mendeley.com/datasets/5y9wdsg2zt/2" "https://com-mendeley-internal.s3.eu-west-1.amazonaws.com/platform/rdm/production/0e960d17-55e7-4814-8338-cb56cdf30fe7?response-content-disposition=attachment%3B%20filename%3D%22Concrete%20Crack%20Images%20for%20Classification.rar%22%3B%20filename%2A%3DUTF-8%27%27Concrete%2520Crack%2520Images%2520for%2520Classification.rar&response-content-type=application%2Fx-rar-compressed&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200131T144729Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAI6DZXOGICLKVGYEA%2F20200131%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Signature=1a259a7a5312492ee03a0ef445e32aa2a15026a17750dd1c4642689c7ff92909" -O "Concrete Crack Images for Classification.rar" -c

!wget --header="Host: digitalcommons.usu.edu" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) coc_coc_browser/85.0.130 Chrome/79.0.3945.130 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7" --header="Referer: https://digitalcommons.usu.edu/all_datasets/48/" --header="Cookie: AMCVS_4D6368F454EC41940A4C98A6%40AdobeOrg=1; bp_plack_session=b0faf45db07af9b1563047ed6a43acb3c4c7e7b2; _ga=GA1.2.452153587.1580464539; _gid=GA1.2.985356459.1580464539; AMCV_4D6368F454EC41940A4C98A6%40AdobeOrg=1075005958%7CMCIDTS%7C18293%7CMCMID%7C75740757516605844704238738647179069886%7CMCAAMLH-1581082817%7C3%7CMCAAMB-1581082817%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1580485217s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.1; __atuvc=5%7C5; __atuvs=5e343db3a53f6014000; _gat_gtag_UA_5499681_18=1; amplitude_id_1d890e80ea7a0ccc43c2b06438458f50usu.edu=eyJkZXZpY2VJZCI6IjA5YzllMzI0LTYzMmYtNDMxYS04MGVlLWNjNjY3YTEwZGY5MVIiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTU4MDQ4MTk3MDg3MSwibGFzdEV2ZW50VGltZSI6MTU4MDQ4MTk3MzQwNCwiZXZlbnRJZCI6NCwiaWRlbnRpZnlJZCI6MCwic2VxdWVuY2VOdW1iZXIiOjR9; s_pers=%20c19%3Dbpdg%253Air_series%253Aarticle%7C1580483771962%3B%20v68%3D1580481971797%7C1580483771971%3B%20v8%3D1580481973993%7C1675089973993%3B%20v8_s%3DLess%2520than%25201%2520day%7C1580483773993%3B; s_sess=%20s_cpc%3D0%3B%20s_ppvl%3Dbpdg%25253Air_series%25253Aarticle%252C55%252C55%252C1068%252C1366%252C668%252C1366%252C768%252C1%252CP%3B%20s_ppv%3Dbpdg%25253Air_series%25253Aarticle%252C55%252C55%252C1068%252C1366%252C668%252C1366%252C768%252C1%252CP%3B%20e41%3D1%3B%20s_cc%3Dtrue%3B%20s_sq%3Delsevier-bpdg-prod%25252Celsevier-global-prod%253D%252526c.%252526a.%252526activitymap.%252526page%25253Dbpdg%2525253Air_series%2525253Aarticle%252526link%25253DSDNET2018.zip%252526region%25253Dbeta_7-3%252526pageIDType%25253D1%252526.activitymap%252526.a%252526.c%252526pid%25253Dbpdg%2525253Air_series%2525253Aarticle%252526pidt%25253D1%252526oid%25253Dhttps%2525253A%2525252F%2525252Fdigitalcommons.usu.edu%2525252Fcgi%2525252Fviewcontent.cgi%2525253Ffilename%2525253D2%25252526article%2525253D1047%25252526context%2525253Dall_datasets%25252526type%252526ot%25253DA%3B" --header="Connection: keep-alive" "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional" -O "SDNET2018.zip" -c
!unzip SDNET2018.zip
!rm SDNET2018.zip
!mkdir data_gray
import cv2 

import glob



files = glob.glob("*")

for f in files :

    if len(f) == 1 :

        file_small = glob.glob(f+"/*")

        for fs in file_small:

            images = glob.glob(fs+"/*")

            for image in images:

                print(image[5:])

                img = cv2.imread(image)

                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

                cv2.imwrite(os.path.join("/kaggle/working/data_gray",image[5:]),gray)

                
# len(os.listdir('data_gray'))
files = glob.glob("/kaggle/input/*")

print(files)

for f in files :

    if len(f) > 1 :

        file_small = glob.glob(f+"/*")

        for fs in file_small:

            images = glob.glob(fs+"/*")

            for image in images:

                print(image.split("/")[5])

                img = cv2.imread(image)

                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

                cv2.imwrite(os.path.join("/kaggle/working/data_gray",image.split("/")[5]),gray)
os.listdir("/kaggle/input")
import zipfile

def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))
zipf = zipfile.ZipFile('/kaggle/working/data.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir('/kaggle/working/data_gray/', zipf)

zipf.close()
!rm -rf /kaggle/working/data_gray

!rm -rf /kaggle/working/P

!rm -rf /kaggle/working/W

!rm -rf /kaggle/working/D