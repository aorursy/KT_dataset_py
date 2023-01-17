import requests
from bs4 import BeautifulSoup

print('Note : Masukkan kata kunci "-i" untuk melihat rincian orang tertentu')
query = input('KeyWord : ').lower()
bol=False
if "-i" in query:
	bol=True
	query = query.replace("-i","")
url = 'https://www.google.com/search?q={}&ie=UTF-8'.format(query)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
source=requests.get(url, headers=headers).text
soup = BeautifulSoup(source, 'html.parser')
try:
	text = soup.find_all("div","rVusze")
	if bol and (text!=[]):
		for x in text:
			print(x.get_text())
	else:
		text2 = soup.find("h2",text="Deskripsi").findNext('span')
		print(text2.get_text())
		if len(text2.get_text())<40:
			print(text2.findNext('span').get_text())
			text2 = text2.findNext('span')
		if len(text2.get_text())<40:
			print(text2.findNext('span').get_text())
			text2 = text2.findNext('span')
		if len(text2.get_text())<40:
			print(text2.findNext('span').get_text())
			text2 = text2.findNext('span')
except:
	try:
		print(soup.find("a","FLP8od").get_text())
	except:
		print("Saya tidak mengerti, semoga membantu ya...\n")
		print(soup.find("span","st").get_text())
