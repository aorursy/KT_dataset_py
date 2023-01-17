import numpy as np # ใส่ค่าimport จากข้อมูลที่มี

import pandas as pd # ประมวลผลข้อมูลdata , CSV file I/O (e.g. pd.read_csv)



#อ่านไฟล์

tsunami_data = pd.read_csv('../input/sources.csv', sep=",", header=0, index_col=0)

tsunami_data.head()
# ตรวจสอบข้อมูล ID, ไม่มี correlatives. 

# แล้วจัดทำข้อมูล reset_index ถูกต้อง



tsunami_data.reset_index(drop=True, inplace=True)

tsunami_data.head()
#ค่าซ้ำและ NaN

#ตรวจสอบหมายเลขของแถว

print("Before cleaning: ",tsunami_data.index)



#Drop and clean row to duplicates

tsunami_data.dropna()

tsunami_data.drop_duplicates()



print("After cleaning: ",tsunami_data.index)



#ไม่มีแถวซ้ำกัน.

#จัดกลุ่มตามประเทศ

tsunami_data['COUNTRY_FRECUENCY'] = tsunami_data.groupby(tsunami_data.COUNTRY)['COUNTRY'].transform('count')
#ข้อมูลสึนามิจากหลายประเทศ

tsunami_data.COUNTRY[tsunami_data.COUNTRY_FRECUENCY > 50].value_counts().plot(kind='bar',

                                             legend=False,

                                             figsize=(12,5),

                                             title="Tsunamis frecuency by countries > 50",

                                             fontsize=12,

                                             alpha=0.5

                                            );
#Tsunamis ความถี่ (year > 1950)

tsunami_data.COUNTRY[(tsunami_data.YEAR >= 1950) & (tsunami_data.COUNTRY_FRECUENCY > 50)].value_counts().plot(kind='bar',

                                             legend=False,

                                             figsize=(12,5),

                                             title="Tsunamis frecuency by countries (Year >= 1950)",

                                             fontsize=12,

                                             alpha=0.5

                                            );
#การทำแผนที่ภูมิภาคที่แตกต่าง

regions = {77:'West Coast of Africa',

            78:'Central Africa',

            73:'Northeast Atlantic Ocean',

            72:'Northwest Atlantic Ocean',

            70:'Southeast Atlantic Ocean',

            71:'Southwest Atlantic Ocean',

            75:'E. Coast USA and Canada, St Pierre and Miquelon',

            76:'Gulf of Mexico',

            74:'Caribbean Sea',

            40:'Black Sea and Caspian Sea',

            50:'Mediterranean Sea',

            30:'Red Sea and Persian Gulf',

            60:'Indian Ocean (including west coast of Australia)',

            87:'Alaska (including Aleutian Islands)',

            84:'China, North and South Korea, Philippines, Taiwan',

            81:'E. Coast Australia, New Zealand, South Pacific Is.',

            80:'Hawaii, Johnston Atoll, Midway I',

            83:'E. Indonesia (Pacific Ocean) and Malaysia',

            82:'New Caledonia, New Guinea, Solomon Is., Vanuatu',

            86:'Kamchatka and Kuril Islands',

            85:'Japan',

            88:'West Coast of North and Central America',

            89:'West Coast of South America'}





tsunami_data['REGIONS'] = tsunami_data['REGION_CODE'].map(regions)



#จัดกลุ่มตามภูมิภาค

tsunami_data['REGIONS_FRECUENCY'] = tsunami_data.groupby(tsunami_data.REGIONS)['REGIONS'].transform('count')
#ความถี่ทางภูมิภาค(ความสูงของคลื่น) ที่สูงกว่า100 โดยใช้ค่า REGIONS มาคำนวน

tsunami_data.REGIONS[tsunami_data.REGIONS_FRECUENCY > 100].value_counts().plot(kind='barh',

                                              legend=False,

                                              figsize=(8,7),

                                              title="Tsunamis by Regions",

                                              alpha=0.5);
#ความถี่ที่เกิดขึ้นต่อปีที่มีความรุ่นแรงที่ค่ามากกว่า 100 โดยใช้ค่า YEAR มาคำนวน

tsunami_data.REGIONS[(tsunami_data.YEAR >= 1916) & (tsunami_data.REGIONS_FRECUENCY > 100)].value_counts().plot(kind='barh',

                                              legend=False,

                                              figsize=(8,7),                                                                                                               

                                              title="Tsunamis by Regions (Year >= 1916)",

                                              alpha=0.5);
#การศึกษาจากประเทศตัวอย่างของ(USA) โดยใช้ค่า REGIONS มาคำนวน

tsunami_data.REGIONS[(tsunami_data.COUNTRY == 'USA')].value_counts().plot(kind='pie',

                                            legend=False,

                                            figsize=(8,7),

                                            title="Tsunamis by Regions (USA)",

                                            autopct='%.1f%%');
#ใส่สาเหตุการเกิดสึนามิที่แตกต่างกัน (0-11) 

causes = {0:'Unknown',

          1:'Earthquake',

          2:'Questionable Earthquake',

          3:'Earthquake and Landslide',

          4:'Volcano and Earthquake',

          5:'Volcano, Earthquake, and Landslide',

          6:'Volcano',

          7:'Volcano and Landslide',

          8:'Landslide',

          9:'Meteorological',

          10:'Explosion',

          11:'Astronomical Tide'}



tsunami_data['CAUSES'] = tsunami_data['CAUSE'].map(causes)



#จัดกลุ่มตามสาเหตุ

tsunami_data['CAUSES_FRECUENCY'] = tsunami_data.groupby(tsunami_data.CAUSES)['CAUSES'].transform('count')
#สาเหตุของสึนามิ โดยใช้ค่า CAUSES มาคำนวน ที่ค่ามากกว่าหรือเท่ากับ 50

tsunami_data.CAUSES[tsunami_data.CAUSES_FRECUENCY >= 50].value_counts().plot(kind='pie',

                                            legend=False,

                                            figsize=(8,7),

                                            title="Causes of tsunamis",

                                            autopct='%.1f%%');
#คลื่นสึนามิของปีตั้งแต่ (year > 1950) โดยใช้ค่า CAUSES ,YEAR ที่เป็นตัวเลขมาคำนวน ตั้งแต่ปี 1950 เป็นต้นไป

tsunami_data.COUNTRY[(tsunami_data.YEAR >= 1950) & (tsunami_data.COUNTRY_FRECUENCY > 50)].value_counts().plot(kind='bar',

                                            legend=False,

                                            figsize=(12,5),

                                            title="Tsunamis frecuency by countries (Year >= 1950)",

                                            fontsize=12,

                                            alpha=0.5

                                            );

#ค่าการคำนวนของการเกิดสาตุที่ได้มาทำเป็นความถี่ที่มาจากสาตุ ที่มีค่าเท่ากับ50 (แต่ละประเทศ)

tsunami_region_mediterranean = tsunami_data[tsunami_data['REGION_CODE'] == 50]

tsunami_region_mediterranean_no_unknown = tsunami_region_mediterranean[(tsunami_region_mediterranean['CAUSES'] != 'Unknown')

                                                                       & (tsunami_region_mediterranean['VALIDITY'] >= 3.0)]

tsunami_mediterranean_cause_group = tsunami_region_mediterranean_no_unknown.groupby(['COUNTRY','CAUSES'])



add_counts = tsunami_mediterranean_cause_group.size().unstack().fillna(0)

normed_subnet = add_counts.div(add_counts.sum(1), axis=0)

add_counts.plot(kind='barh', legend=True, figsize=(11,7), stacked=True);
#ใช้ค่าปีมาคำนวนด้วยกับค่าที่ได้จากสาเหตุ คำนวนจะได้ค่าที่เป็นความถี่ที่เป็นปี ที่เริ่มตั้งแต่ปี 1800 ขึ้นไป

tsunami_region_mediterranean_no_unknown_1800 = tsunami_region_mediterranean_no_unknown[(tsunami_region_mediterranean_no_unknown['YEAR'] >= 1800)

                                                                                      & (tsunami_region_mediterranean_no_unknown['VALIDITY'] >= 3.0)]

tsunami_region_mediterranean_no_unknown_1800_group = tsunami_region_mediterranean_no_unknown_1800.groupby(['COUNTRY','CAUSES'])



add_counts = tsunami_region_mediterranean_no_unknown_1800_group.size().unstack().fillna(0)

normed_subnet = add_counts.div(add_counts.sum(1), axis=0)

add_counts.plot(kind='barh', legend=True, figsize=(11,7), stacked=True);
#สรุป แสองข้อมูลที่ได้จากการคำนวนทั้งหมด ที่เป็นค่าความถี่ที่เกิดสินามิทั้งหมด

tsunami_region_japan_1700 = tsunami_data[(tsunami_data['YEAR'] >= 1700) 

                                         & (tsunami_data['REGION_CODE'] == 85) 

                                         & (tsunami_data['CAUSES'] != 'Unknown')

                                         & (tsunami_data['VALIDITY'] >= 3.0)]

tsunami_region_japan_1700.count()

tsunami_region_japan_1700_group = tsunami_region_japan_1700.groupby(['DAMAGE_TOTAL','CAUSES'])



add_counts = tsunami_region_japan_1700_group.size().unstack().fillna(0)

normed_subnet = add_counts.div(add_counts.sum(1), axis=0)

add_counts.plot(kind='bar', legend=True, figsize=(11,7), stacked=True);
#tsunami_data.EVENT_VALIDITY.value_counts()