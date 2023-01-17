!pip install requests
 !pip install beautifulsoup4
import requests
html = requests.get("https://www.baruch.cuny.edu/")

html
content= html.content.decode("utf-8")

title = content[content.find("<title>"):]

print(title)
from bs4 import BeautifulSoup
html_doc = """

<html><head><title>The Dormouse's story</title></head>

<body>

<p class="title"><b>The Dormouse's story</b></p>



<p class="story">Once upon a time there were three little sisters; and their names were

<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,

<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and

<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;

and they lived at the bottom of a well.</p>



<p class="story">...</p>

</body>

<html>

"""
soup = BeautifulSoup(html_doc, 'html.parser')
soup.prettify()
soup.title
## TODO: Another example of a tag

soup.a
soup.title.name
soup.a.name
soup.title.string
p = soup.p

p
p.attrs
p.attrs["class"]
soup = BeautifulSoup(html_doc, 'html.parser')
head_tag = soup.head

head_tag
head_tag.contents
list(head_tag.children)
list(soup.body.children)
list(head_tag.children)[0].parent
all_anchors = soup.find_all('a')
all_anchors
soup.find_all('a')
all_anchors[1].attrs
all_anchors[0].get("href")
all_anchors[1].string
for a in all_anchors:

    print(a.get("href"))