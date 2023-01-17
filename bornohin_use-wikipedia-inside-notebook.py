# First thing first. Lets install wikipedia module and import it.

!pip install wikipedia

import wikipedia

print("\n\n\nWikipedia is ready for use.")
# Its easy to find certain topics.

page_capitals = wikipedia.page("List of national capitals")

# So what type of objects are we dealing with?

type(page_capitals)
# This should have the url we need.

page_capitals.url
# now that we have the URL to work with, lets grab the contents.

page_content = page_capitals.content

print(type(page_content))  # This tells us the object type.

print(page_capitals.title)  # This should bring up correct page title. 
print(page_content)
page_capitals.categories
capture_the_flags = page_capitals.images

# Lets check for any inconsistencies.

capture_the_flags
print(f"We have flags of total {len(capture_the_flags)} countries here.")

!pip install wget

import wget

print("\n\n\n\nwget is ready for use")
# We will make a list, capture the names of the flags as well as download them in working directory.

flags = []



# Looping to get the first 5 flags only.



for i in capture_the_flags[:5]:

    # Use wget download method to download specified image url.

    image_filename = wget.download(i)



    print('Image Successfully Downloaded: ', image_filename)

    flags.append(image_filename)
flags
# import os

# os.remove("./Flag_of_Afghanistan.svg")  # file name