# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read the html content from the url link 'https://en.wikipedia.org/wiki/Python_(programming_language)'. 

#Store the content in variable html_content.

#Create a BeautifulSoup object with html_content and html.parser. Store the result in variable soup.

#Find the number of reference links present in soup object. Store the result in variable n_links.

#Hint : Make use of find_all method and look of a tags.

#print n_links
#url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'

#from urllib import request

#html_content = request.urlopen(url).read()

#from bs4 import BeautifulSoup

#soup = BeautifulSoup(html_content,'html.parser')

#n_links = len(soup.find_all('a'))

#print(n_links)
#Find the table from soup object, having class attribute value wikitable. Store the result in variable table.

#Hint: Make use of find method associated with soup object.

#Find all rows of table. Store the result in rows.

#Hint: Make use of find_all method on table and look for tr tags.

#Ignore the first row of rows, with expression rows = rows[1:]

#For every row perform the following tasks.

#Find all columns associated with a row. Hint: Make use of find_all on each row and look for td tags.

#Select only the first column from the obtained list of columns.

#print the text associated with first column. Hint: Make use of get_text method on the obtained column.
from bs4 import BeautifulSoup

html2 = '''<table class="wikitable">

<caption>Summary of Python 3's built-in types

</caption>

<tbody><tr>

<th>Type

</th>

<th>mutable

</th>

<th>Description

</th>

<th>Syntax example

</th></tr>

<tr>

<td><code>bool</code>

</td>

<td>immutable

</td>

<td><a href="/wiki/Boolean_value" class="mw-redirect" title="Boolean value">Boolean value</a>

</td>

<td><code>True</code><br /><code>False</code>

</td></tr>

<tr>

<td><code>bytearray</code>

</td>

<td>mutable

</td>

<td>Sequence of <a href="/wiki/Byte" title="Byte">bytes</a>

</td>

<td><code>bytearray(b'Some ASCII')</code><br /><code>bytearray(b"Some ASCII")</code><br /><code>bytearray([119, 105, 107, 105])</code>

</td></tr>

<tr>

<td><code>bytes</code>

</td>

<td>immutable

</td>

<td>Sequence of bytes

</td>

<td><code>b'Some ASCII'</code><br /><code>b"Some ASCII"</code><br /><code>bytes([119, 105, 107, 105])</code>

</td></tr>

<tr>

<td><code>complex</code>

</td>

<td>immutable

</td>

<td><a href="/wiki/Complex_number" title="Complex number">Complex number</a> with real and imaginary parts

</td>

<td><code>3+2.7j</code>

</td></tr>

<tr>

<td><code>dict</code>

</td>

<td>mutable

</td>

<td><a href="/wiki/Associative_array" title="Associative array">Associative array</a> (or dictionary) of key and value pairs; can contain mixed types (keys and values), keys must be a hashable type

</td>

<td><code>{'key1': 1.0, 3: False}</code>

</td></tr>

<tr>

<td><code>ellipsis</code>

</td>

<td>

</td>

<td>An <a href="/wiki/Ellipsis_(programming_operator)" class="mw-redirect" title="Ellipsis (programming operator)">ellipsis</a> placeholder to be used as an index in <a href="/wiki/NumPy" title="NumPy">NumPy</a> arrays

</td>

<td><code>...</code>

</td></tr>

<tr>

<td><code>float</code>

</td>

<td>immutable

</td>

<td><a href="/wiki/Floating_point" class="mw-redirect" title="Floating point">Floating point</a> number, system-defined precision

</td>

<td><code>3.1415927</code>

</td></tr>

<tr>

<td><code>frozenset</code>

</td>

<td>immutable

</td>

<td>Unordered <a href="/wiki/Set_(computer_science)" class="mw-redirect" title="Set (computer science)">set</a>, contains no duplicates; can contain mixed types, if hashable

</td>

<td><code>frozenset([4.0, 'string', True])</code>

</td></tr>

<tr>

<td><code>int</code>

</td>

<td>immutable

</td>

<td><a href="/wiki/Integer_(computer_science)" title="Integer (computer science)">Integer</a> of unlimited magnitude<sup id="cite_ref-pep0237_79-0" class="reference"><a href="#cite_note-pep0237-79">&#91;79&#93;</a></sup>

</td>

<td><code>42</code>

</td></tr>

<tr>

<td><code>list</code>

</td>

<td>mutable

</td>

<td><a href="/wiki/List_(computer_science)" class="mw-redirect" title="List (computer science)">List</a>, can contain mixed types

</td>

<td><code>[4.0, 'string', True]</code>

</td></tr>

<tr>

<td><code>set</code>

</td>

<td>mutable

</td>

<td>Unordered <a href="/wiki/Set_(computer_science)" class="mw-redirect" title="Set (computer science)">set</a>, contains no duplicates; can contain mixed types, if hashable

</td>

<td><code>{4.0, 'string', True}</code>

</td></tr>

<tr>

<td><code>str</code>

</td>

<td><a href="/wiki/Immutable_object" title="Immutable object">immutable</a>

</td>

<td>A <a href="/wiki/Character_string" class="mw-redirect" title="Character string">character string</a>: sequence of Unicode codepoints

</td>

<td><code>'Wikipedia'</code><br /><code>"Wikipedia"</code><br /><code>"""Spanning<br />multiple<br />lines"""</code>

</td></tr>

<tr>

<td><code>tuple</code>

</td>

<td>immutable

</td>

<td>Can contain mixed types

</td>

<td><code>(4.0, 'string', True)</code>

</td></tr></tbody></table>'''



soup = BeautifulSoup(html2)
table = soup.find("table", attrs={"class":"wikitable"})
rows = [row for row in table.find_all('tr')][1:]
for row in rows:

    columns = row.find_all('td')[0]

    print(columns.get_text())