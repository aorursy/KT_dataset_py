import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
regions = ["Bishkek", "Chui", "Issyk-Kul", "Naryn",
           "Batken", "Batket", "Jalal-Abad", "Jalalabat", 
           "Talas", "Osh"]
loans_kyrgyzstan = pd.read_csv('../input/kiva_loans.csv')
loans_kyrgyzstan = loans_kyrgyzstan.loc[loans_kyrgyzstan.country == 'Kyrgyzstan']
file = pd.read_csv('../input/kiva_loans.csv', usecols = ["country", "region", "use"])
cities = ["Balykchy", "Batken", "Bishkek", "Bordunskiy", "Cholpon-Ata", "Cholponata", 
          "Gulcha", "Isfana", "Jalal-Abad", "Jalalabad", "Kadamjay", "Kaindy", "Kant", 
          "Kara-Balta", "Karabalta", "Karakol", "Kara-Suu", "Karasuu", "Kemin", 
          "Kerben", "Ketmen'tebe","Ketmentebe", "Khaidarkan", "Kochkor-Ata", "Kochkorata",
          "Kok-Janggak", "Kokjanggak","Kok-Tash", "Koktash", "Kyzyl-Jar", "Kyzyljar",
          "Kyzyl-Kiya", "Kyzylkiya", "Mailuu-Suu", "Mailuusuu", "Naryn", "Nookat",
          "Orlovka", "Orto-Toyok", "Ortotoyok", "Osh", "Pristan'-Przheval'sk",
          "Pristanprzhevalsk", "Shamaldy-Say", "Shamaldysay", "Shopokov",
          "Sulukta", "Talas", "Tash-Komur", "Tashkomur", "Tokmok", "Toktogul","Uzgen",
          "Vostochny"]
urban_loans = loans_kyrgyzstan[loans_kyrgyzstan['region'].str.contains('|'.join(cities), na=False) & ~loans_kyrgyzstan['region'].str.contains('village', na=False)]
urban_loans
subFile = file[(file["country"].str.contains("Kyrgyzstan")==True)]
dicty = {}
allReg = {
    "Bishkek":0, 
    "Chui":0,
    "Issyk-Kul":0,
    "Naryn":0,
    "Batken":0,      
    "Jalal-Abad":0, 
    "Talas":0,
    "Osh (city)": 0,
    "Osh":0
}

for x in subFile["region"]:
    if "Bishkek" in str(x): allReg["Bishkek"]+=1
    elif "Chui" in str(x): allReg["Chui"]+=1
    elif "Issyk-Kul" in str(x): allReg["Issyk-Kul"]+=1
    elif "Naryn" in str(x): allReg["Naryn"]+=1
    elif "Batken" in str(x) or "Batket" in str(x): allReg["Batken"]+=1
    elif "Jalalabad" in str(x) or "Jalalabat" in str(x): allReg["Jalal-Abad"]+=1
    elif "Talas" in str(x): allReg["Talas"]+=1
    elif "Osh region" in str(x): allReg["Osh"]+=1
    elif "Osh" in str(x): allReg["Osh (city)"]+=1


for x in subFile["region"]:
    if any(y in str(x) for y in regions): continue
    else: dicty[x] = dicty.get(x, 0) + 1
for x in allReg: print(x, allReg[x])
poor_dist = {
    "Bishkek":8, 
    "Chui":7,
    "Issyk-Kul":14,
    "Naryn":7,
    "Batken":5,
    "Jalal-Abad":24,
    "Talas":6,
    "Osh (city)": 30,
    "Osh":30
}

non_mount_regions = ["Karabalta", "Kara-Balta", "Bishkek", "Tokmok", "Talas", "Kirov",
                     "Tash-Komur", "Tashkomur", "Jalal-Abad", "Jalalabad", "Osh", 
                     "Kyzyl-Kiya", "Kyzylkiya"]
non_mountainous_loans = loans_kyrgyzstan[loans_kyrgyzstan['region'].str.contains('|'.join(non_mount_regions), na=False)]
non_mountainous_loans
children_loans = loans_kyrgyzstan[loans_kyrgyzstan["use"].str.contains("her child|her children|his child| his children", na=False)]
children_loans
not_poor_cities_or_region = ["Bishkek", "Chuy", "Chui"]
low_poverty_regional_loans = loans_kyrgyzstan[loans_kyrgyzstan['region'].str.contains('|'.join(not_poor_cities_or_region), na=False)]
low_poverty_regional_loans
poor_cities_or_regions = ["Issyk-Kul", "Issykkul", "Cholpon-Ata", "Cholponata", "Karakol",
                          "Kyzyl-Suu", "Kyzylsuu", "Balykchy", "Kara-Koo", "Karakoo",
                          "Tosor", "Bosteri", "Bokonbaev", "Tup"]
high_poverty_regional_loans = loans_kyrgyzstan[loans_kyrgyzstan['region'].str.contains('|'.join(poor_cities_or_regions), na=False)]
high_poverty_regional_loans
urban_positive = loans_kyrgyzstan.copy(deep=True)
urban_positive["urban"] = loans_kyrgyzstan.isin(urban_loans)['id']
urban_positive
non_mount_positive = loans_kyrgyzstan.copy(deep=True)
non_mount_positive["non_mount"] = loans_kyrgyzstan.isin(non_mountainous_loans)['id']
non_mount_positive
children_positive = loans_kyrgyzstan.copy(deep=True)
children_positive["children"] = loans_kyrgyzstan.isin(children_loans)['id']
children_positive
high_poverty_region_positive = loans_kyrgyzstan.copy(deep=True)
high_poverty_region_positive["high_poverty_region"] = loans_kyrgyzstan.isin(high_poverty_regional_loans)['id']
high_poverty_region_positive
final_loans = urban_positive.copy(deep=True)
final_loans["non_mount"] = non_mount_positive["non_mount"]
final_loans["children"] = children_positive["children"]
final_loans["high_poverty_region"] = high_poverty_region_positive["high_poverty_region"]
final_loans