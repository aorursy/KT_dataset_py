import pandas as pd
import requests
from bs4 import BeautifulSoup



c1=[]
c2=[]
c3=[]


    
url2 = 'https://kocsiguru.hu/elado-hasznalt-autok'
response = requests.get(url2)
soup = BeautifulSoup(response.text, 'html.parser')
    
   
    #vételár
results1 = soup.find_all('p', attrs={'class': 'hidden price'})
    #név
results2 = soup.find_all('h3', attrs={'class':'text-dark-royal-blue name'})
    #üzemanyag
results3 = soup.find_all('li', attrs={'class':'fuel'})

    
for i in range(len(results1)):
        text=results1[i].get_text().strip()
        c1.append(text)
for i in range(len(results2)):
        text=results2[i].get_text().strip()
        c2.append(text)    
for i in range(len(results3)):
        text=results3[i].get_text().strip()
        c3.append(text)
   

tabla_autok={'Vételár':c1,
                'Neve':c2,
                'Üzemanyag':c3}
df_autok = pd.DataFrame.from_dict(tabla_autok)


writer = pd.ExcelWriter('autok.xlsx', engine='xlsxwriter')
df_autok.to_excel(writer, sheet_name='Használt autók', index=False)
writer.save()

