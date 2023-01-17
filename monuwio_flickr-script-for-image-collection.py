from flickrapi import FlickrAPI
KEY = 'Your_key'
SECRET = 'Your_secret'
SIZES = ["url_o", "url_k", "url_h", "url_l", "url_c"]  # order of preference

def get_photos(image_tag):
    extras = ','.join(SIZES)
    flickr = FlickrAPI(KEY, SECRET)
    photos = flickr.walk(text=image_tag,
                            extras=extras,  # get the url for the original size image
                            privacy_filter=1,  # search only for public photos
                            per_page=100,
                            sort='relevance')
    return photos

def get_url(photo):
    for i in range(len(SIZES)):
        url = photo.get(SIZES[i])
        if url:  # if url is None try with the next size
            return url

def get_urls(image_tag, max):
    photos = get_photos(image_tag)
    counter=0
    urls=[]

    for photo in photos:
        if counter < max:
            url = get_url(photo)  # get preffered size url
            if url:
                urls.append(url)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break

    return urls
import requests
import os
import sys

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def download_images(urls, path):
    create_folder(path)  # makes sure path exists

    for url in urls:
        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                outfile.write(response.content)
all_species = ['cars exposition inside']
images_per_specie = 1000

def download():
    for specie in all_species:

        print('Getting urls for', specie)
        urls = get_urls(specie, images_per_specie)

        print('Downlaing images for', specie)
        path = os.path.join('data', specie)

        download_images(urls, path)
start_time = time.time()

download()

print('Took', round(time.time() - start_time, 2), 'seconds')