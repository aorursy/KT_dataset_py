# Scrape a webpage by Natalie Winkel



import requests

import time

import re



url = 'http://wordcruncher.com/search-tools.html'

headers = {'Natalie Winkel': 'nwinkel1206@gmail.com'}

response = requests.get(url, headers=headers)



html = response.text



with open('wordcruncher.html', 'w', encoding='utf-8') as fOut:

    print(html, file=fOut)

# Remove all new lines to put everything on one line.

page_source = re.sub(r'\n', '', html)



divs = re.findall(r'<div class="[^"]+?" role="[^"]+?">.+?</div>', page_source)



# Clean up all of the strings to remove tags, extra space.

for i in divs:

    i = re.sub(r'<[^>]+?>', '', i)

    i = re.sub(r'\s\s+?', '', i)

    print(i)

    with open('wordcruncher_parsed.txt', 'w', encoding='utf-8') as fOut:

        print(i, file=fOut)

        

print('An HTML and TXT file have been saved!')
