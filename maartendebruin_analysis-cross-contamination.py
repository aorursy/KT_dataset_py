import pandas as pd

data = pd.read_csv('../input/Kruisbesmetting.csv')
del data['respondent_id'] 
del data['collector_id'] 
del data['date_created'] 
del data['date_modified'] 
del data['ip_address'] 
del data['email_address'] 
del data['first_name'] 
del data['last_name'] 

data.rename(columns={'Vul een van de boxen in per allergie, dus geef aan of u allergisch, intolerant of niet allergisch bent': 'Melk-A',}, inplace=True)
data.rename(columns={'Unnamed: 10': 'Melk-I', 'Unnamed: 11': 'Melk-NA'}, inplace=True)
data.rename(columns={ 'Unnamed: 15': 'Gluten-A', 'Unnamed: 16': 'Gluten-I', 'Unnamed: 17': 'Gluten-NA'}, inplace=True)
data.rename(columns={'Unnamed: 12': 'Ei-A', 'Unnamed: 13': 'Ei-I', 'Unnamed: 14': 'Ei-NA'}, inplace=True)
data.rename(columns={'Unnamed: 18': 'Noten-A', 'Unnamed: 19': 'Noten-I', 'Unnamed: 20': 'Noten-NA'}, inplace=True)

data.rename(columns={'Heeft u weleens een allergische reactie ervaren in een restaurant?(meerdere antwoorden mogelijk)': 'Ja-Ja'}, inplace=True)
data.rename(columns={'Unnamed: 22': 'Ja-Nee', 'Unnamed: 23': 'Ja-Idk', 'Unnamed: 24': 'Nee-Nee'}, inplace=True)

data.rename(columns={'Heeft u weleens een allergische reactie ervaren zonder te weten wat de oorzaak is?': 'Oorzaak-Onbekend', 'Unnamed: 26': 'Oorzaak-Bekend' }, inplace=True)

data.rename(columns={'Heeft u weleens een allergisch reactie ervaren van een product met de volgende tekst op het label? (100% zeker dat de allergische reactie van dit specifieke product komt)Vul een van de boxen in per allergie, dus laat weten of u nooit, soms of altijd een allergische reactie ervaart van een product met de onderstaande tekst op het label.': 'Vrij-Nooit',}, inplace=True)
data.rename(columns={'Unnamed: 28': 'Vrij-Soms','Unnamed: 29': 'Vrij-Vaak'}, inplace=True)
data.rename(columns={'Unnamed: 30': 'Sporen-Nooit', 'Unnamed: 31': 'Sporen-Soms','Unnamed: 32': 'Sporen-Vaak'}, inplace=True)
data.rename(columns={'Unnamed: 33': 'Fabriek-Nooit', 'Unnamed: 34': 'Fabriek-Soms','Unnamed: 35': 'Fabriek-Vaak'}, inplace=True)

data.fillna(0, inplace=True)

print (data.info())
data.fillna(0, inplace=True)

data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Ja-Ja']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Ja-Nee']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Ja-Idk']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Oorzaak-Bekend']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Oorzaak-Onbekend']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Vrij-Vaak']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Sporen-Vaak']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Fabriek-Vaak']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Vrij-Soms']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Sporen-Soms']).size().reset_index().rename(columns={0:'count'})
data.groupby(['Melk-A', 'Melk-I', 'Ei-A', 'Ei-I', 'Gluten-A', 'Gluten-I','Noten-A', 'Noten-I','Fabriek-Soms']).size().reset_index().rename(columns={0:'count'})