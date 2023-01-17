# Prog-08: Web Scraping

# 6330011321 : กฤติพงศ์ มานะชำนิ

# ...

# ข้าพเจ้า นายกฤติพงศ์ มานะชำนิ เขียนโค้ดนี้ด้วยตัวเอง โดยได้แรงบันดาลใจจากการจำกัดเขตพื้นที่ค้นหาไปเรื่อยๆจาก html tag และ class

# You can follow more about this code performance in https://www.kaggle.com/neospirit/prog08-web-scraping

import urllib

import urllib.request as urq



def load_html(page_url):

    return str(urq.urlopen(page_url).read().decode('utf-8'))

#-------------------------------------------------

def get_faculty_names(url):

    faculty_names = []

    full_html = load_html(url).split("\n")

    

    for line in full_html:

        if '<h3 class="text-title-1">' in line:

            a = line.split(">")

            faculty = a[2].split("<")[0]

            if "Faculty of" in faculty: faculty_names.append(faculty)

    

    return faculty_names



def save_image(img_url, filename):

    urq.urlretrieve(img_url, filename)

  

def download_faculty_images(url):

    full_html = load_html(url).split('<div class="post-media">')[1:-1]

    

    for line in full_html:

        checking = line.split("\n")

        

        for i in checking:

            

            if '<h3 class="text-title-1">' in i:

                a = i.split(">")

                faculty = a[2].split("<")[0]

                

                if "Faculty of" in faculty: 

                    component = line.split('srcset')

                    

                    for j in component:

                        if '300x188' in j:

                            link = j.split(",")[1]

                            link = link.strip(" \n\t\t\t")

                            link = link.strip("300w")                                         # Example: https://waiiinta.github.io/image/chula-faculty-allied-health-sciences-hero-desktop-300x188.jpg 

                            

                            imagefile = link.split("https://waiiinta.github.io/image/")[1]    # Example: chula-faculty-allied-health-sciences-hero-desktop-300x188.jpg 

                            

                            save_image(link, imagefile)

                            break

  



def print_faculty_numbers(url):

    full_html = load_html(url).split('<div class="post-media">')[1:-1]

    faculty_list = get_faculty_names(url)

    

    for component in full_html:

        

        split_component = component.split(">")[0]

        split_component = split_component.strip("\n\t\t\t<a href=")

        split_component = split_component.strip('"')

        

        faculty_web = load_html(split_component)

        

        faculty_name = faculty_web.split("title")[1]

        faculty_name = faculty_name.split("–")[0][1:-1]

        

        if faculty_name in faculty_list:

            faculty_contact = faculty_web.split("<strong>Contact</strong>")[1]

            faculty_contact = faculty_contact.split("<strong>Tel:</strong>")[1]

            faculty_contact = faculty_contact.split("<br>")[0]

            faculty_contact = faculty_contact.split("+")[1][0:16]

            if "(0) " in faculty_contact:

                faculty_contact = faculty_contact.replace("(0) ","")

            faculty_contact = faculty_contact.replace("66", "0")

            faculty_contact = faculty_contact[0:11]

            

            faculty_name = faculty_name.replace(" ","-").lower()

            print(faculty_name)

            print(faculty_contact)

            print("\n")

#-------------------------------------------------

def main():

    pageurl = "https://waiiinta.github.io/"



    print(get_faculty_names(pageurl))



    download_faculty_images(pageurl)



    print_faculty_numbers(pageurl)

#-------------------------------------------------

main()
