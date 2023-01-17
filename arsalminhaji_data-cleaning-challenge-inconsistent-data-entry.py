import  pandas  as  pd 
import  numpy  as  np

import  fuzzywuzzy
from fuzzywuzzy import  process 
import  chardet 
np.random.seed(0)
with  open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv" ,"rb") as rawdata:
    result=chardet.detect(rawdata.read(100000))
#sucide_killing_data=pd.read_csv(r"C:\Users\Imran\Downloads\PakistanSuicideAttacks Ver 11 (30-November-2017).csv")
print(result)

sucide_killing_data=pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv" ,  encoding='Windows-1252')
sucide_killing_data
cities=sucide_killing_data['City'].unique()
cities.sort()
cities
sucide_killing_data['City']=sucide_killing_data['City'].str.lower()
sucide_killing_data['City']=sucide_killing_data['City'].str.strip()
sucide_killing_data['City']
province_data=sucide_killing_data["Province"].unique()
province_data
sucide_killing_data['Province']=sucide_killing_data["Province"].str.lower()
sucide_killing_data['Province']=sucide_killing_data['Province'].str.strip()
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
def replace_matches_in_column(data_frame,column,string_to_match,min_ratio=90):
    strings=data_frame[column].unique()
    print(strings)
    matches=fuzzywuzzy.process.extract(string_to_match, strings , limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    print(matches)
    close_match=[matches[0] for matches in matches if matches[1] >= 90]
    print(close_match)
    rows_with_matches=data_frame[column].isin(close_match)
    print(rows_with_matches)
    data_frame.loc[rows_with_matches,column] =string_to_match
    print('All done')
replace_matches_in_column(data_frame=sucide_killing_data , column='City', string_to_match="d.i khan")
check_Dg_khan=sucide_killing_data['City'].unique()
check_Dg_khan.sort()
check_Dg_khan
replace_matches_in_column(data_frame=sucide_killing_data,column='City' ,string_to_match="kuram agency")
check_Dg_khan=sucide_killing_data['City'].unique()
check_Dg_khan.sort()
check_Dg_khan