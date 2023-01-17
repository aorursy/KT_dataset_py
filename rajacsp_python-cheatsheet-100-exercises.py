# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')
# 01. Multiple arguments



def print_everything(*args):

    for index, item in enumerate(args):

        print(f'{index}. {item}')

        

print_everything('Toronto', 'Montreal', 'New York', 'Waterloo')
# 02. Article Reader



import requests

from bs4 import BeautifulSoup



# Collect and parse first page

page = requests.get('https://www.wired.com/story/inside-twitter-hack-election-plan/')

soup = BeautifulSoup(page.text, 'html.parser') 



content = soup.select('div.article__body')[0]



print(content.get_text()[:100])
# 03. Add Padding Around String



text = 'Hey Kaggle'



# Add Spaces Of Padding To The Left

print('\n', format(text, '>40'))



# Add Spaces Of Padding To The Right

print('\n', format(text, '<40'))



# Add Spaces Of Padding On Each Side

print('\n', format(text, '^40'))



# Add * Of Padding On Each Side

print('\n', format(text, '*^40'))
# 04. Class 2 JSON



import json



class City:

    def toJSON(self):

        return json.dumps(self, default = lambda o: o.__dict__, sort_keys = True, indent = 4)

    

me = City()

me.name = "Toronto"

me.id = 102



print(me.toJSON()) 
# 05. Collections Counter



from collections import Counter



cnt = Counter()



for word in ['Toronto', 'Montreal', 'Montreal', 'Waterloo', 'Toronto', 'Toronto']:

    cnt[word] += 1

    

print(cnt)



print(type(cnt))
# 06. Compress Image



from PIL import Image



# Open the image

im = Image.open("/kaggle/input/numpy-cheatsheet/cn_tower.jpg")



# im.save("/kaggle/input/numpy-cheatsheet/cn_tower_1.jpg", format = "JPEG", quality = 70) # This will save
# 07. Compress Image CV2



from PIL import Image

import cv2



# Open the image

img = cv2.imread('/kaggle/input/numpy-cheatsheet/cn_tower.jpg')



# save image with lower quality—smaller file size

# cv2.imwrite('/kaggle/input/numpy-cheatsheet/cn_tower_1.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 0])
# 08. Copy List



import copy



a = [1, 2, 3]

print(a)



b = copy.copy(a)

print(b)



b.append(4)

print(b)



c = copy.deepcopy(a)

print(c)
# 09. Current Directory



import os



dirpath = os.getcwd()

print("current directory is : " + dirpath)



foldername = os.path.basename(dirpath)

print("Directory name is : " + foldername)
# 10. DNS Lookup



import socket



addr1 = socket.gethostbyname('google.com')

addr2 = socket.gethostbyname('yahoo.com')



print('google.com : ', addr1)

print('yahoo.com  : ', addr2)
# 11. Date Util



import datetime



now = datetime.datetime.now()



print('now.year : ', now.year)



print ('now.strftime("%y") : ', now.strftime("%y"))



print ('now.strftime("%d") : ', now.strftime("%d"))



print ('now.strftime("%m") : ', now.strftime("%m"))



print('datetime.datetime.today() : ', datetime.datetime.today())
# 12. Date Util 2



import datetime



cur_date = datetime.date.today()

current_year = cur_date.year # Extract current year only

current_month = cur_date.month # Extract current month only

current_day = cur_date.day # Extract current day only



print('current date : ', cur_date)



print('current month : ', current_month)



print('current day : ', current_day)



current_date = cur_date.strftime("%d-%m-%Y")

print('current date with format : ', current_date)



current_date = cur_date.strftime("%d-%B-%Y")

print('current date with format 2 : ', current_date)
# 13. Default Dict



from collections import defaultdict



user = defaultdict(lambda: 'Kevin')



print('Default Dictionary : ')

print(user)



print('\nDictionary by random key : ')

print(user['abc'])



print('\nDictionary by another random key : ')

print(user['country'])



user['name'] = 'Peter'

user['age']  = 21

user['city'] = 'Toronto'



print('\nDictionary after assigning value : ')

print(user)



print('\nDictionary by available key (city) : ')

print(user['city'])



# iterate dictionary

print('\nIterating Default dictionary')

for k, v in user.items():

    print(k, "==>", v)
# 14. Enumerate



data = ['Toronto', 'Montreal', 'Waterloo', 'Chennai', 'Bangalore', 123, ['One', 'Two', 'Three']]



for x in enumerate(data):

    print(x)

    

for i, x in enumerate(data):

    print(i, "==>", x)
# 15. Enumerate with index



# start the index from 200

for i, x in enumerate(data, 200):

    print(i, "==>", x)
# 16. Enumerate List



animals = ['cat', 'dog', 'monkey']



print('enumerate with string:')

for idx, animal in enumerate(animals):

    print ('#%d: %s' % (idx , animal))

    

print('\nenumerate with f-strings:')

for idx, animal in enumerate(animals):

    print (f'#{idx}: {animal}')  
# 17. Fibonacci



def fib(n):

    if n == 0:

        return 0

    elif n == 1:

        return 1

    else:

        return fib(n-1) + fib(n-2)

    

print(fib(8))

print(fib(13))
# 18. Fibonacci in Generator



def fibonacci(n):

    a = b = 1

    

    for i in range(n):

        yield a

        a, b = b, a+b

    

for x in fibonacci(10):

    print(x)
# 19. Fibonacci Lambda



def print_fibo(number):

    print(list(map(lambda x, f = lambda x, f : (f(x-1,f) + f(x-2,f)) if x > 1 else 1: f(x,f), range(number))))

    

print_fibo(6)
# 20. Find Domain



def get_domain(url):

    spltAr = url.split("://")

    i = (0,1)[len(spltAr)>1]

    dm = spltAr[i].split("?")[0].split('/')[0].split(':')[0].lower()

    

    #print(dm)

    return dm



print(get_domain('http://www.oreilly.com/people/adam-michael-wood'))
# 21. Find Domain without prefix



def get_domain_without_prefix(url):

    spltAr = url.split("://")

    i = (0,1)[len(spltAr)>1]

    dm = spltAr[i].split("?")[0].split('/')[0].split(':')[0].lower()

    

    if("www." in dm):

        dm = dm.replace("www.", "")

    

    return dm



print(get_domain_without_prefix('https://javabeat.net/introduction-to-hibernate-caching/'))
# 22. Find Python/Conda Env Location



import sys



print(sys.executable)
# 23. Function With Return Type



def square(a:int) -> int:

    return a * a



print(square(4))



def square_1(a:int) -> str:

    return a * a



print(type(square_1(15))) # this is not working as expected. Need to verify this again
# 24. Geo Location 1



import requests

import json 



send_url = 'http://api.ipstack.com/50.100.30.136?access_key=49ad529d309a09477749245782d260b8&format=1'



r = requests.get(send_url)

j = json.loads(r.text)



lat = j['latitude']

lon = j['longitude']



print(lat)

print(lon)
!pip install geocoder
# 25. Geo Location by Geocoder



import geocoder



# g = geocoder.ip('me')

# print(g.latlng)  # this will show the current user ip



test_location = geocoder.ip('151.101.1.69')

print(test_location.latlng)
# 26. Get System Attribute



import sys



print(getattr(sys, 'version'))
# 27. Get Default Encoding



import sys



print(sys.getdefaultencoding())
# 28. TBW



!pip install ipwhois
# 29. IP Whois Simple



from ipwhois import IPWhois

from pprint import pprint



obj = IPWhois('133.1.2.5')

results = obj.lookup_whois(inc_nir = True)



pprint(results)
# 30. IP To Long



import math



class AuthException(Exception):

    pass



def ip_to_long(ip_address):

    

    if(ip_address == '0:0:0:0:0:0:0:1'):

        ip_address = '127.0.0.1'

        

    ip_address_array = ip_address.split('.')

    

    if(len(ip_address_array) != 4):    

        raise AuthException('Invalid Ip')

    

    #print(ip_address_array[1])

    

    num = 0

    for i in range(len(ip_address_array)):

        #print(ip_address_array[i])

        power = 3 - i

        num = num + ( (int(ip_address_array[i]) % 256 * int(math.pow(256, power)) ))

    

    return num



ip_long = ip_to_long('10.3.81.34')        

print(ip_long)
!pip install xmltodict
# 31. JSON to XML



import xmltodict



content = {

  "note" : {

    "to" : "Tove",

    "from" : "Jani",

    "heading" : "Reminder",

    "body" : "Don't forget me this weekend!"

  }

}



print(content)



xml = xmltodict.unparse(content, pretty=True)



print(xml)
# 32. Lambda Custom



def is_south_indian_city(city):

    if city == 'chennai' or city == 'madurai' or city == 'bengaluru':

        return True

    

    return False



cities = [

    'chennai', 'delhi', 'madurai', 'pune', 'bengaluru'

]



new_list = list(filter(lambda x: is_south_indian_city(x) , cities))

print(new_list)
# 33. Lambda Function: Odd



def is_odd(digit):

    if digit % 2 != 0:

        return True

    

    return False 



digits = [

    1, 7, 18, 2, 4, 2, 8, 5, 3

]



new_list = list(filter(lambda x: is_odd(x) , digits))



print(new_list)
# 34. Square calculation by Lambda



squares = list(map(lambda x: x**2, range(10)))



print(squares)
# 35. Linear Search



def linear_search(lys, element):  

    for i in range (len(lys)):

        if lys[i] == element:

            return i

    return -1



list = [1, 2, 5, 6]



print(linear_search(list, 5))
# 36. List Operations



list = [

    "AB", 

    "CD"

]



print('Basic list:')

print(list)



# append to list

list.append("ABC")

print('\nafter appending ABC:')

print(list)



list.extend(["EFG", "IJK", "LMN"])

print('\nafter extending 3 string list:')

print(list)



list.extend(("OPQ", "RST"))

print('\nafter extending tuple:')

print(list)



list.extend(range(1, 5))

print('\nextending range:')

print(list)



list2 = ["One", "Two"]

print(list2)



list.append(list2)

print('\nafter appending list:')

print(list)



list3 = ["Four", "Five"]



list += list3

print('\nconcat list:')

print(list)



print(list.pop(len(list)-1))

print('\nafter popping last element:')

print(list)



list.remove('AB')

print('\nafter remving a string:')

print(list)
# 37. Loop a Dictioary



a_dict = {'person': 2, 'cat': 4, 'spider': 8}



for animal in a_dict:

    legs = a_dict[animal]

    print ('A %s has %d legs' % (animal, legs))
!pip install magicattr
# 38. Class with Magicattr



import magicattr



class City:

        

    def __init__(self, name, country):

        self.name = name

        self.country = country

        

to = City('Toronto', 'Canada')

ch = City('Chennai', 'India')



print(to)



print(magicattr.get(to, 'country'))
# 39. Modify String in Place



import io



s = "Toronto is awesome"



sio = io.StringIO(s)

print(sio.getvalue())



sio.seek(11)

print(sio.write("Wonderful"))



print(sio.getvalue())
# 40. More Itertools



import itertools as it

import more_itertools as mit



a = it.count(0, 2)

mit.take(10, a)
# 41. Named Arguments in Format



print("{greeting!r:20}".format(greeting = "Hello"))



print("{one} {two!r}".format(one = "ten", two = "twenty"))
# 42. Python Path



import sys



print(sys.path)
# 43. Quick Sorting



def quicksort(arr):

    

    if len(arr) <= 1:

        return arr

    

    pivot  = arr[len(arr) // 2]

    left   = [x for x in arr if x < pivot]

    middle = [x for x in arr if x == pivot]

    right  = [x for x in arr if x > pivot]

    

    return quicksort(left) + middle + quicksort(right)



print(quicksort([3,6,8,10,1,2,1]))
# 44. Random Binary



import random



MAX = 5



for x in range(MAX):

    val = random.randint(0, 1)

    print(val)
# 45. Random Timestamp



from random import randrange

import datetime



def random_date(start, l):

    

    current = start

    

    while l >= 0:

        curr = current + datetime.timedelta(minutes=randrange(60))

        yield curr

        l -= 1

    

def get_random_timestamp(max):

    

    startDate = datetime.datetime(2013, 9, 20,13, 0)



    for x in random_date(startDate, max-1):

        print(x.strftime("%d/%m/%y %H:%M"))



        print(x)

        

        timestamp = datetime.datetime.timestamp(x)

        print(int(timestamp))

        

        print('-' * 20)

        

get_random_timestamp(3)
# 46 Range with Intervals



for x in range(0, 10, 2):

    print(x)

    

print('-' * 10)

    

for x in range(0, 100, 10):

    print(x)
# 47. Read GitHub CSV



from io import BytesIO

import requests

import pandas as pd



FILEPATH = 'https://raw.githubusercontent.com/rajacsp/public-dataset/master/zynicide-wine-reviews/winemag-data-130k-v2.csv'



r = requests.get(FILEPATH)

data = r.content



onlinedata = BytesIO(data)



df = pd.read_csv(onlinedata, index_col=0)



df.head()
# 48. Read GitLab CSV



from io import BytesIO

import requests

import pandas as pd



filename = 'https://gitlab.com/rajacsp/datasets/raw/master/FEELI%20SONG%20LIST%20-%20Songs.csv'



r = requests.get(filename)

data = r.content



df = pd.read_csv(BytesIO(data), index_col=0)



df.head()
# 49. Read JSON Online



import urllib.request, json



with urllib.request.urlopen("https://jsonplaceholder.typicode.com/todos/1") as url:

    data = json.loads(url.read().decode())

#     print(data['tweets'][0])

    print(data)

    print(data['userId'])
# 50. Read Online CSV



from io import BytesIO

import requests

import pandas as pd



filename = 'https://gitlab.com/rajacsp/datasets/raw/master/amazon_co_ecommerce_sample.csv'



r = requests.get(filename)

data = r.content



df = pd.read_csv(BytesIO(data), index_col=0)



df.head()
# 51. Read Online CSV with Unicode Escape



from io import BytesIO

import requests

import pandas as pd





filename = 'https://www.sample-videos.com/csv/Sample-Spreadsheet-10-rows.csv'



r = requests.get(filename)

data = r.content



df = pd.read_csv(BytesIO(data), index_col = 0, encoding = 'unicode_escape')



df.head()
# 52. Reduce



from functools import reduce



def add(a, b): 

    return a + b



# add(10, 15)



reduce(add, [10, 20, 30, 40])
# 53. Reduce Advanced



from functools import reduce



def do_magic(a, b): 

    

    if((a % 2 == 1) & (b % 2 == 1) ):

        return a * b

    

    return a + b



reduce(do_magic, [10, 20, 30, 40])
# 54. Sum with Reducers



import functools as fnt



def get_sum(list):

    return fnt.reduce(lambda x,y : x+y, list)



list = [10, 40, 20, 50, 30, 90]



print(get_sum(list))
# 55. Remove Item from list



names = [

    "Kevin",

    "Peter",

    "John"

]



print(names)



names.pop(0)



print(names)
# 56. Remove Item on Loop



list = [

    1, 2, 3, 4, 5, 6

]



for i, j in enumerate(list):

    if(j % 2 == 0):

        del list[i]

        

print(list)
# 57. Remove Item with filterfalse



list = [1, 2, 3, 4, 5, 6, 7, 8]

from itertools import filterfalse



list[:] = filterfalse((lambda x: x%3), list)

print(list)
# 58. Remove Item with Short form



list = [1, 2, 3, 4, 5, 6, 7, 8]

new_list = [x for x in list if x % 2 == 1]



print(new_list)
# 59. Remove th From Date



from datetime import datetime

import re



content = 'Sunday, May 18th, 2019'



d = datetime.strptime(re.sub('(\d+)(st|nd|rd|th)', '\g<1>', content), '%A, %B %d, %Y')



print(d)
# 60. Sort Dictionary By Value



dict = {

    'one' : 8,

    'two' : 1,

    'three' : 12

}



print(dict)



import operator



sorted_x = sorted(dict.items(), key=operator.itemgetter(1))



print(sorted_x)
# 61. Dictionary sort with collections



import collections



sorted_dict = collections.OrderedDict(dict)



# Reverse sort

reverse_dict= sorted(dict.items(), key=lambda x: x[1], reverse=True)



print(reverse_dict)



for x in reverse_dict:

    print(x[0], "==>", x[1])
# 62. Sort Map By Key



def sort_by_key(dict):

    

    for key in sorted(dict.keys()):

        print("%s: %s" % (key, dict[key]))

    

    return None



age = {

    'carl':40,

    'alan':2,

    'bob':1,

    'danny':3

}



sort_by_key(age)
# 63. Sort Map by Value



def sort_by_value(dict):



    age_sorted = sorted(dict.items(), key=lambda x:x[1])

       

    return age_sorted



age = {

    'carl':40,

    'alan':2,

    'bob':1,

    'danny':3

}



age_sorted = sort_by_value(age)



print(age_sorted)
# 64. Split with Multiple



import re



content = "Hey, you - what are you doing here!?"



contents = re.findall(r"[\w']+", content)



for c in contents:

    print(c)
# 65. Static on the fly



class Mathematics:



    def addNumbers(x, y):

        return x + y

    

# create addNumbers static method

Mathematics.add = staticmethod(Mathematics.addNumbers)



print('The sum is:', Mathematics.add(5, 10))
# 66. Statistics Simple



import statistics



data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]



print(statistics.mean(data))



print(statistics.median(data))



print(statistics.variance(data))
# 67. String Startswith



content = "Toronto is awesome"



print(content.startswith("Toronto"))
# 68. Sys Version



import sys



print(sys.version)
# 69. Text Diff



import difflib



str1 = "I understand how customers do their choice. Difference"

str2 = "I understand how customers do their choice."



seq = difflib.SequenceMatcher(None, str1, str2)



d = seq.ratio() * 100

print(d)
# 70. Text Similarity



def get_similarity(str1, str2):

    seq = difflib.SequenceMatcher(None, str1, str2)

    d = seq.ratio()*100

    return d



get_similarity("Toronto is nice", "Toronto is looking nice")
# 71. Timeit Simple



import timeit

import numpy as np



def add(x, y):

    return x + y



repeat_number = 100000

e = timeit.repeat(

    stmt='''add(a, b)''', 

    setup='''a=2; b=3;from __main__ import add''', 

    repeat=3,

    number=repeat_number

)



print('Method: {}, Avg.: {:.6f}'.format("eta", np.array(e).mean()))
# 72. Timeit Simple 2



import timeit



def some_function():

    return map(lambda x: x^2, range(10))



time1 = timeit.timeit(some_function)

print(time1)
# 73. Try Catch with Custom Error



class NegativeMarksError(Exception):

   pass



def check_marks(mark):

    try:

        

        if(mark < 0):

            raise NegativeMarksError

        

        if(mark > 50):

            print('pass')

        else:

            print('fail')

    except ValueError:

        print('ValueError: Some value error')

    except TypeError:

        print('TypeError: Cannot convert into int')

    except NegativeMarksError:

        print('Negative Marks are not allowed')

        

check_marks(10)



check_marks('10')



check_marks(-10)
# 74. Try Catch with Raise Error



def check_marks(mark):

    try:

        

        if(mark < 0):

            raise ValueError

        

        if(mark > 50):

            print('pass')

        else:

            print('fail')

    except ValueError:

        print('ValueError: Some value error')

    except TypeError:

        print('TypeError: Cannot convert into int')

        

check_marks('10')



check_marks(-10)
# 75. Letter Match



import re



contents = [

    "AB23",

    "A2B3", 

    "DE09",

    "MN90",

    "XYi9",

    "XY90"

]



for content in contents:

    

    regex_pattern = "(?:AB|DE|XY)\d+" 

    # starts with AB or DE; should contain one or more number



    m = re.match(regex_pattern, content)



    if(m):

        print('matched : '+content)
# 76. Regex Simple Match



import re



content = "Eminem means positive"



a = re.search('^Em.*ve$', content)



if(a):

    print('matched')

else:

    print('not matched')
# 77. Regex and Index



import re



pattern = 'awesome'



content = 'Duckduck go is awesome and it is getting better everyday'



match = re.search(pattern, content)



#print(match)



start = match.start()

end = match.end()



# print(match.string)



print(start, end)
# 78. Compile Expressions



import re



# precompile regex patterns

regex_entries = [

    re.compile(p)

    for p in ['awesome', 'ocean']

]



content = 'Duckduck go is awesome and it is getting better everyday'



print('Text : {!r}'.format(content))



for regex in regex_entries:

    print('Finding  {} -> '.format(regex.pattern), end = ' ')



    if(regex.search(content)):

        print('matched')

    else:

        print('not matched')
# 79. Regex - Find Multiple Whitespace



import re



print(re.split(r'\s{2,}', "2012.03.04       check everything      status: OK"))





content = "2012.03.04       check everything      status: OK"



list = re.split(r'\s{2,}', content)



for a in list:

    print(a)
# 80. Regex - Find Two Digits with Spaces



import re



text = "It happened on Feb 21 at 3:30"



answer= re.findall(r'\b\d{2}\b', text)

print(answer)



# To match only two digits with spaces

answers = re.findall(r'\s\d{2}\s', text)



for a in answers:

    print(a.strip())
# 81. Regex - Get Full Matches



import re

from typing import List



_RGX = re.compile(r'(.)\1*')

def long_repeat(string: str) -> List[str]:

    return [m.group(0) for m in _RGX.finditer(string)]



print(long_repeat('devvvvveeeeeeeeeeeloooooooooper'))



print(long_repeat('country'))
# 82. Regex - Get Specific Contents



content = """<!DOCTYPE html>



  <!-- The following setting enables collapsible lists -->

  <p>

  <a href="#human">Human</a></p>



  <p class="collapse-section">

  <a class="collapsed collapse-toggle" data-toggle="collapse" 

  href=#mammals>Mammals</a>

  <div class="collapse" id="mammals">

  <ul>

  <li><a href="#alpaca">Alpaca</a>

  <li><a href="#armadillo">Armadillo</a>

  <li><a href="#sequence_only">Armadillo</a> (sequence only)

  <li><a href="#baboon">Baboon</a>

  <li><a href="#bison">Bison</a>

  <li><a href="#bonobo">Bonobo</a>

  <li><a href="#brown_kiwi">Brown kiwi</a>

  <li><a href="#bushbaby">Bushbaby</a>

  <li><a href="#sequence_only">Bushbaby</a> (sequence only)

  <li><a href="#cat">Cat</a>

  <li><a href="#chimp">Chimpanzee</a>

  <li><a href="#chinese_hamster">Chinese hamster</a>

  <li><a href="#chinese_pangolin">Chinese pangolin</a>

  <li><a href="#cow">Cow</a>

  <li><a href="#crab-eating_macaque">Crab-eating_macaque</a>

  <div class="gbFooterCopyright">

  &copy; 2017 The Regents of the University of California. All 

  Rights Reserved.

  <br>

  <a href="https://genome.ucsc.edu/conditions.html">Conditions of 

  Use</a>

  </div>"""



import re



regex = r"<li.+?#[^s].+?>(.+)?<\/.+>"



find_matches = re.findall(regex, content)

for matches in find_matches:

    print(matches)
# 83. Regex - Ontario Postal Code



import re



names = ['M2N1H5', 'M2N 1H5', 'M882J8']



regex_patten = "^[A-Z]\d[A-Z]\s*\d[A-Z]\d"

# starts with Capital letter; it can have zero or one space



for name in names:

    m = re.match(regex_patten, name)



    if(m):

        #print(m.groups())

        print('matched : ', name)
# 84. Regex - Remove Numbers



import re



content = '15. TFRecord in Tensorflow'



result = re.sub(r'\d+\.', '', content)



print(result.strip())
# 85. Regex - Remove Numbers 2



content2 = """1. Cross Entropy Loss Derivation



2. How to split a tensorflow model into two parts?



3. RNN Text Generation with Eager Execution



4. Softmax activation in Tensor"""



result2 = re.sub(r'\d+\.', '', content2)



# result2.strip()



for c in result2.strip().split('\n'):

    print(c.strip())
# 86. Regex - Remove Symbols



import re



content = "This device is costing only 12.89$. This is fantastic!"



result = re.sub(r'[^\w]', ' ', content)



print(result)
# 87. Regex - Search



import re



data = [

    "name['toronto'] = 'One'",

    "state['ontario'] = 'Two'",

    "country['canada']['ca'] = 'Three'",

    "extra['maple'] = 'Four'"

]



REGEX = re.compile(r"\['(?P<word>.*?)'\]")



for line in data:

    found = REGEX.search(line)

    if found:

        print(found.group('word'))
# 88. Class Basics



class Employee:

    def __init__(self, name, age, salary):

        self.name = name

        self.age = age

        self.salary = salary

        

    def __repr__(self):

        return repr((self.name, self.age, self.salary))

    

employees = [

    Employee('Peter', 21, 6),

    Employee('Kevin', 22, 4),

    Employee('Simon', 21, 8)

]



print(employees)



# sorting

print(sorted(employees, key = lambda e: e.salary))



print(sorted(employees, key = lambda e: -e.salary))
# 89. Sort by Attrgetter



from operator import itemgetter, attrgetter



print(sorted(employees, key=attrgetter('age')))



young_employees = sorted(employees, key=attrgetter('age'), reverse=False)

print(young_employees)
# 90. Fuzzy String Ratio



from fuzzywuzzy import fuzz



print(fuzz.ratio("this is a test", "this is a test!"))



print(fuzz.ratio("this is a cup", "this is a world cup"))
# 91. List Op - sublist from to



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[4:8])
# 92. List Op - Get first 4



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[:4])
# 93. List Op - from index to last



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[4:])
# 94. List Op - Get all as subset



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[:])
# 95. List Op - Get last item



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[-1])
# 96. List Op - Get last second item



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[-2])
# 97. List Op - Get last 2 items



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# get last 2 items

print(a[-2:])
# 98. List Op - Get every items except last 2 items



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# every except last 2 iteme

print(a[:-2])
# 99. List Op - Get all item reversed



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# all items reverse

print(a[::-1])
# 100. List Op - Get first 2 items reversed



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



print(a[1::-1])
# 101. List Op - Get first 3 items reversed



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# get the first three items reversed

print(a[2::-1])
# 102. List Op - Get all items reversed and ignore the last item



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# all items eversed and ignore the last item

print(a[-2::-1])
# 103. List Op - Remove last 2 item, reverse the rest



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# Remove last 2 items, reverse all the rest of the items

print(a[-3::-1])
# 104. List Op - Get last item reversed



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# last two items reversed

print(a[:-3:-1])
# 105. List Op - Everything except the last 3 items, reversed



a = [

    10, 20, 30, 40, 50, 60, 70, 80, 90

]



# Everything except the last 3 items, reversed

print(a[-4::-1])
# 105. String format



a = "x{} y{} z{}"



b = a.format(10, 20, 30)



print(b)
# 106. Unpack tuple in String format



a = "x{} y{} z{}"

b_tuple = (11, 21, 31)



c = a.format(*b_tuple)



print(c)



# Caution

# d = a.format(b_tuple) ==> This will throw :: IndexError: tuple index out of range
# 107. Unpack list in String format



a = "x{} y{} z{}"

b_list = [11, 21, 31]



# print(type(b_list))



c = a.format(*b_list)



print(c)



# Caution

# d = a.format(b_tuple) ==> This will throw :: IndexError: tuple index out of range
# 107. Unpack list in String format



a = "x{} y{} z{}"

b_dict = {

    'a' : 10,

    'b' : 20

}



# c = a.format(*b_dict) # This will throw IndexError: tuple index out of range
# 108. Get the second slowest number in the list



a = [1, 23, 23, 1, 89, 55, 78, 23, 12, 17, 17, 34, 34, 6]



print('Original list:')

print(a)



b = sorted(set(a))[1]

print('\nSecond smallest number : ')

print(b)
# 109. Combine list elements by index



import itertools



a = [10, 20, 30]

b = [1, 2, 3]

c = [19, 29, 39]



print('Original:')

print('\na:')

print(a)

print('\nb:')

print(b)

print('\nc:')

print(c)



d = []

for x,y,z in zip(a,b,c):

    d.extend((x,y,z))



print('\nCombined list:')

print(d)
# 110. Find in JSON String



import json



jsonString ='''[{"city":"Toronto","result":{"dummy":"twenty","_code":"to"}},

{"city":"Montreal","result":{"dummy":"one","_code":"mo"}},

{"city":"New York","result":{"dummy":"elevel","_code":"ny"}}]'''



json_content = json.loads(jsonString)

print('Original:')

print(json_content)



print('\nFind _code:')

for index in range(len(json_content)):

    content = json_content[index]["result"]['_code']

    print(content)
# 111. Word to Chars



word = "Toronto"



letters = list(word)



for l in letters:

    print(l)
!pip install phonenumbers
# 112. Get Region Code for phone numbers



import phonenumbers



from phonenumbers.phonenumberutil import (

    region_code_for_country_code,

    region_code_for_number,

    country_code_for_region

)



pn = phonenumbers.parse('+1 647 898 3434')

print('Region code for Phone Number +1 647 898 3434:')

print(region_code_for_number(pn))



print('\nRegion Code for Country Code 91:')

print(region_code_for_country_code(91))



print('\nCountry Code for Region:')

print(country_code_for_region('IN'))
# 113. Get Country and Codes



import pycountry



canada = pycountry.countries.get(alpha_2='CA')



print('Canada:')

print(canada)

print(canada.name)



us = pycountry.countries.get(alpha_2 = 'US')

print('\nUSA:')

print(us)
# 114. Number counter with defaultdict



from collections import defaultdict



counter = defaultdict(int)



content = "one 2 1 three 4 2 1 7 6 six 8 9 8"



for c in content.split(' '):

    if(c.isdigit()):

        counter[c] += 1

    

for k, v in counter.items():

    print(k, "==>", v)
# 115. 



class Person:  



    def __init__(self, name, job, pay):

        self.job  = job

        self.pay  = pay



    def giveRaise(self, percent):

        self.pay = int(self.pay + (self.pay * percent)/100)

        

mark = Person('Mark', 'developer', 100000)

print('Pay before raise:')

print(mark.pay)



mark.giveRaise(5)



print('\nPay after raise:')

print(mark.pay)
# 116. Locals of Function



def myFunc():

    print("hello")

    

fname = "myFunc"



f = locals()[fname]

f()



f = eval(fname)

f()