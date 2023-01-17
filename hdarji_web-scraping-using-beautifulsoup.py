import requests
from bs4 import BeautifulSoup
import pandas as pd


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26228-syracuse-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



r= pd.DataFrame(temp)
r.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
r
df=r.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
df

df = df[df.Name != 'Welcome to Yocket']
p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
p.head()
columns = ['GRE', 'Undergrad']
df.drop(columns, inplace=True, axis=1)
df
df.to_csv("Syracuse1.csv")
pd.read_csv("Syracuse1.csv")
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/95-university-of-maryland-college-park/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.head()
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p
p = p[p.Name != 'Welcome to Yocket']
p
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("Umcp.csv")
pd.read_csv("Umcp.csv")
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/310-university-of-illinois-at-chicago/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p = p[p.Name != 'Welcome to Yocket']
p.head()
p.to_csv("Uic1.csv")
pd.read_csv("Uic1.csv")
#14-carnegie-mellon-university
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/46196-carnegie-mellon-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p = p[p.Name != 'Welcome to Yocket']
p.head()
p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
p.head()
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("CMU.csv")
pd.read_csv("CMU.csv")
#14-carnegie-mellon-university
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/14-carnegie-mellon-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p = p[p.Name != 'Welcome to Yocket']
p.head()
p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("CMU2.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/46121-university-of-washington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p = p[p.Name != 'Welcome to Yocket']
p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("UWash.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/46140-university-of-washington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p = p[p.Name != 'Welcome to Yocket']
p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("Uwash2.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/49-texas-a-and-m-university-college-station/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)
p = p[p.Name != 'Welcome to Yocket']
p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("TAMU.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/368-indiana-university-bloomington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("kelley.csv")
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/158-university-of-cincinnati/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)
p.to_csv("UCinn.csv")
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26891-university-of-texas-dallas/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)



p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UTD.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/913-northeastern-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)
p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("NEU.csv")
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26239-university-of-arizona/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Eller.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1417-rutgers-university-new-brunswick/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers1.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/423-university-of-minnesota-twin-cities/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("minnesota.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/545-georgia-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("gsu.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/816-worcester-polytechnic-institute/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("wpi.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/461-rensselaer-polytechnic-institute/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("RPI.csv")


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/6-boston-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("BU.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1043-university-of-california-irvine/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UCI.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/128-state-university-of-new-york-at-buffalo/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("SunyB.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/318-northwestern-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("northwestern.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/253-arizona-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("asu.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/77-university-of-florida/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Uflorida.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/363-university-of-north-carolina-at-charlotte/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Uncc.csv")


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1451-stevens-institute-of-technology/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Stevens.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1249-university-of-iowa/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UIowa.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/179-rochester-institute-of-technology/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("RIT.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/349-university-of-texas-arlington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UT_arlington.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/622-santa-clara-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("SantaClara.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/414-illinois-institute-of-technology/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("IIT.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/402-university-of-delaware/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("delaware.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26240-university-of-utah/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UUtah.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1272-drexel-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("drexel.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/104-university-of-texas-austin/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UT_Austin.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/589-new-york-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("NYU.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1063-pennsylvania-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Penn_State.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/278-university-of-pennsylvania/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UPenn.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/326-iowa-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Iowa_State.csv")

temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/57-university-of-california-los-angeles/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UCLA.csv")

