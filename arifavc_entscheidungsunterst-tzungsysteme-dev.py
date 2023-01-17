import pandas as pd

import json

import requests

from pandas import json_normalize

from matplotlib import pyplot as plt
response = requests.get('https://corona-api.com/countries').json()['data']
covid = json_normalize(response)
covid.shape
covid.head(10)
covid.drop(['updated_at','coordinates.latitude','coordinates.longitude','latest_data.calculated.recovered_vs_death_ratio',],

           axis = 1, inplace=True)
covid.head()
data = pd.read_csv('../input/country-mapping-iso-continent-region/continents2.csv')
regions = pd.DataFrame(data)
regions.head()
cols_to_keep = ['alpha-2','region']
regions = regions[cols_to_keep]
regions.head()
regions.rename(columns = {'alpha-2':'code'}, inplace = True)
data = covid.merge(regions , on='code')
data.sort_values(by="latest_data.confirmed", ascending=False, inplace=True)
data.reset_index(inplace=True)
del data['index']
data
data.dropna(subset=['latest_data.calculated.death_rate'],inplace=True)
data.shape
data["region"].value_counts()
grouped_data = data.groupby(["region"])
grouped_data.get_group("Asia")
continent = grouped_data.get_group("Asia").sort_values(by="latest_data.confirmed", ascending=False)
continent.head()
def alpha_2(country_name):

        with open('../input/alpha2/ulkeler.json' ,encoding="utf8") as f:

            data = json.load(f)

        for alpha_code,name in data.items():

            if name.lower() == country_name.lower():

                code = alpha_code

                return code

        if country_name[0] == "i" or country_name[0] == "I" :

            raise NameError(f'{country_name} ülkesinin ilk harfini şu şekilde yazın => {"İ"+country_name[1:]}')

        else:

            raise NameError(f'{country_name} değerine karşılık gelen bir alpha-2 kodu bulunamadı')
def timeline_graph(arg="World"):

        if arg == 'World':

            response = requests.get('https://corona-api.com/timeline').json()['data']

        else:

            code = alpha_2(arg)

            response = timeline = requests.get(f'https://corona-api.com/countries/{code}').json()['data']['timeline']

            

        timeline = response

        data = {'dates':[],'confirmed':[],'deaths':[],'recovered':[]}

        for day in timeline:

            data['dates'].append(day['date'])

            data['confirmed'].append(day['confirmed'])

            data['deaths'].append(day['deaths'])

            data['recovered'].append(day['recovered'])

                  

   

        plt.rcParams["figure.figsize"] = [16,9]

        plt.style.use("fast")

        plt.plot(data.get('dates')[::-1],data.get('confirmed')[::-1],  label = "confirmed")

        plt.plot(data.get('dates')[::-1],data.get('deaths')[::-1],  label = "deaths")

        plt.plot(data.get('dates')[::-1],data.get('recovered')[::-1],  label = "recovered")





        

        plt.xlabel("Tarih")

        

        plt.xticks(rotation='vertical', fontsize='8')

        plt.ylabel("Sayı")

        plt.tight_layout()

        plt.title(f"Covid19 {arg} Timeline")



        plt.legend()



        plt.show()



    
timeline_graph()
timeline_graph("Türkiye")
timeline_graph("Amerika")
def confirmed_graph(kita):

    plt.rcParams["figure.figsize"] = [16,9]

    plt.style.use("seaborn-ticks")

    continent = grouped_data.get_group(kita).sort_values(by="latest_data.confirmed", ascending=False)



    plt.barh(continent.iloc[0:15]['name'], continent.iloc[0:15]['latest_data.confirmed'],  label = "# of case")



    plt.xlabel("Cases")

    plt.ylabel("Country")

    plt.title(f"Covid19 {kita} verileri")



    plt.legend(loc='upper right')

    plt.grid(True)



    plt.tight_layout()



    plt.show() 

    
confirmed_graph("Asia")
confirmed_graph("Europe")
def graph(kita,data,log=False):

    plt.rcParams["figure.figsize"] = [16,9]

    plt.style.use("seaborn-ticks")

    continent = grouped_data.get_group(kita).sort_values(by=f"latest_data.{data}", ascending=False)



    plt.barh(continent.iloc[0:15]['name'], continent.iloc[0:15][f'latest_data.{data}'],  label = f"# of {data}", log=log)



    plt.xlabel("Cases")

    plt.ylabel("Country")

    plt.title(f"{kita} | {data}")



    plt.legend(loc='upper right')

    plt.grid(True)



    plt.tight_layout()



    plt.show() 

    
graph("Asia","deaths")
graph("Europe",'confirmed')
graph("Americas","deaths")
graph("Americas","deaths",log=True)
def piechart(kita,data):

    continent = grouped_data.get_group(kita).sort_values(by=f"latest_data.{data}", ascending=False)

    plt.rcParams["figure.figsize"] = [25,16]

    plt.style.use("fast")

    



    plt.title(f"{kita} | {data}")

    



    plt.pie(continent.iloc[0:10][f'latest_data.{data}'], labels =continent.iloc[0:10]['name'] , shadow=True,

            startangle=70, autopct="%2.1f%%", 

            wedgeprops={"edgecolor":"black"})



    plt.tight_layout()



    plt.show( )
piechart("Asia","confirmed")
piechart("Europe","deaths")