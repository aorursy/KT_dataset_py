import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nutrition_code as nut # nutrition module



SOURCE_DIR = '../input/nutrition/'

nut.load(source=SOURCE_DIR)
usda = nut.tables['nutrition']

usda[[x for x in usda.columns if not x in ['nutrition id','crop group']]].head()
foodbal = nut.tables['diets_global']

years = list(foodbal.year.unique())

countries = list(foodbal.country.unique())

foodbal.head()
pd.pivot_table(foodbal,values='amount kg_pa',columns='year',index='country',aggfunc=np.sum)
food_item = nut.tables['food_item']

food_item[[x for x in food_item.columns if not x == 'yield']].head()
china_modern = nut.Diet.from_country(2013,'China')

china_modern.nutrition()
diet_index = [(x,y) for x in years for y in countries]

diets = [nut.Diet.from_country(x[0],x[1]) for x in diet_index]

food_groups = [d.food_groups() for d in diets]

yr = pd.DataFrame({'year':[x[0] for x in diet_index]})

cntry =pd.DataFrame( {'country':[x[1] for x in diet_index]})

energy = pd.DataFrame({'energy':[fg['01 cereal refined']+fg['05 animal'] for fg in food_groups]})

nut_rich = pd.DataFrame({'nutrient rich':[sum(fg)-fg['01 cereal refined']-fg['05 animal'] 

                                  for fg in food_groups]})

nut_avg = np.mean(nut_rich)



fgrcds = pd.concat([yr,cntry,energy/nut_avg['nutrient rich'],nut_rich/nut_avg['nutrient rich']],axis=1)

fgrcds.set_index(['year','country'],inplace=True)

fgrcds.loc[(1965,)]
nut_basis = [d.nutrition() for d in diets]

fat_pr = pd.DataFrame({'fat_pr':[nt['fats']+nt['protein'] for nt in nut_basis]})

carb_fib = pd.DataFrame({'carb_fib':[nt['carbs']+nt['fiber'] for nt in nut_basis]})



cf_avg = np.mean(carb_fib)



ntrcds = pd.concat([yr,cntry,fat_pr/cf_avg.carb_fib,carb_fib/cf_avg.carb_fib],axis=1)

ntrcds.set_index(['year','country'],inplace=True)

ntrcds.loc[(1965,)]
fgrcds.loc[(2013,)]/fgrcds.loc[(1965,)]-1
ntrcds.loc[(2013,)]/ntrcds.loc[(1965,)]-1
sgp = nut.sgp_diet()

sgp.food_groups()
sgp.nutrition()
cho_rcds = nut.tables['cho']

cho_pvt = pd.pivot_table(cho_rcds,columns='nutrition group',index='element',values='wt')

cho_pvt.loc[['C','H','O','N','S']][['carbs','mono unsaturated','polyunsaturated',

        'sat fats','protein','fiber']]
sgp.elements()
sgp.cn_ratio()