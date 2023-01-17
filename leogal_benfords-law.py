import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os
os.listdir('../input')
df = pd.read_csv('../input/articles1.csv')
df.head(5)
numbers = []
for i, row in df.iterrows():
    article_txt = row['content']
    title_txt = row['title']
    
    article_numbers = re.findall(r'[1-9][0-9]*', article_txt)
    title_numbers = re.findall(r'[1-9][0-9]*', title_txt)
    
    numbers.extend(article_numbers)
    numbers.extend(title_numbers)
first_digit = {}
second_digit = {}

for num in numbers:
    try:
        first_digit[num[0]] += 1
    except KeyError:
        first_digit[num[0]] = 1
    
    try:
        try:
            second = num[1]
        except IndexError:
            continue
        
        second_digit[num[1]] += 1
    except KeyError:
        second_digit[num[1]] = 1
plt.figure(figsize=(20, 7))

plt.subplot(131)
plt.bar([num for num in first_digit], [first_digit[num] for num in first_digit])
plt.title('Occurences of first digits')
plt.xlabel('First digit')
plt.ylabel('Occurences')

plt.subplot(132)
plt.bar([num for num in second_digit], [second_digit[num] for num in second_digit])
plt.title('Occurences of second digits')
plt.xlabel('Second digit')
plt.ylabel('Occurences')

plt.subplot(133)
plt.bar([num for num in range(1, 10)], [math.log10((num + 1) / num) for num in range(1, 10)])
plt.title('Benfords Law')
plt.xlabel('First digit')
plt.ylabel('Distribution')