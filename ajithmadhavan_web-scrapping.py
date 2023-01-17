import urllib.request as url

from bs4 import BeautifulSoup
page = url.urlopen("http://www.estesparkweather.net/archive_reports.php?date=200701")
soup=BeautifulSoup(page,'lxml')

# print(soup.find_all('table')

months=soup.find_all('form')

ddData=[]

for item in months:

    #print(item.find_all('option'))

    for i in item.find_all('option'):

        ddData.append(i.get('value'))

#print(ddData)

ddData=ddData[::-1]

print(ddData)

params=[]

#for dd in ddData:

    #print(arrow.get(dd,'YYYYD').date())

    
import arrow

def get_date(para,day):

    dt=para+day

    a=arrow.get(dt,'YYYYMD').date()

    return str(a)

monthlyData=[]

def createDataFrame(data,param,month_text):

    for content in data:

        dailyData=[]

        

        for items in content:

            if items.startswith(month_text):

                day=items.split()[1]

                if len(day)==1:

                    day='0'+day

                if day!='00':

                    dailyData.append(get_date(param,day))

                else:

                    break

                #print(dailyData)

            if items.startswith('Average temperature'):

                avgTemp=items.split()[1].split('temperature')[1]

                dailyData.append(avgTemp)

            if items.startswith('Average humidity'):

                avgHum=items.split()[1].split('humidity')[1]

                dailyData.append(avgHum)

            if items.startswith('Average dewpoint'):

                avgDew=items.split()[1].split('dewpoint')[1]

                dailyData.append(avgDew)

        monthlyData.append(dailyData)

            



month_list=['Jan', 'Feb' ,'Mar', 'Apr' ,'May' ,'Jun', 'Jul','Aug' ,'Sep','Oct','Nov','Dec']



for param in ddData[:20]:

    #print(param)

    page1=url.urlopen("http://www.estesparkweather.net/archive_reports.php?date="+param)

    #print(page1)

    soup1=BeautifulSoup(page1,'lxml')

    str3=soup1.find_all('table')

    dataTemp=[]

    data=[]

    for items in str3:

        item=items.text.splitlines()

        dataTemp.append(item)

    #print("dataTemp=",dataTemp)

    index=param[4:]

    finalIndex=int(index)-1

    for item in dataTemp:

        while '' in item:

            item.remove('')

        

    for x in dataTemp:

        if x[0].startswith(month_list[finalIndex]):

            data.append(x)

    #print(month_list[finalIndex])

    createDataFrame(data,param,month_list[finalIndex])

#print(data)

        

if len(monthlyData)>0:

    print(monthlyData)