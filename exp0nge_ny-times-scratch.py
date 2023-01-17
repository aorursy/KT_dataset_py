import requests

from bs4 import BeautifulSoup
my_url = "https://www.nytimes.com/section/technology"
doc = requests.get(my_url)

doc
# doc.content
my_soup = BeautifulSoup(doc.content)

# my_soup
articles = my_soup.find_all("li", class_="css-ye6x8s")
for article in articles:

#     print(article.find("h2").string)

    print(article.find_all("div", class_="css-1lc2l26"))

    print(article.find_all("time"))
for article in articles:

#     print("article")

    childrens = list(article.find("a").children)

    child = childrens[2]

    print(child.text)
articles = my_soup.find_all("li", class_="css-ye6x8s")
type(articles)
len(articles)
my_list = list(articles[2].children)

my_list
def add_two(a, b):

    return a + b
def add_two_2(a, b):

    print(a + b)
my_var = add_two(1, 2)

print("test")

print(my_var)
add_two_2(1, 2)

print("test")
def my_attrs():

    return [1, 2, 3]