import pandas as pd
import numpy as np

from nose.tools import *
world_food_data=pd.read_csv("../input/world-food-facts/en.openfoodfacts.org.products.tsv", sep="\t", low_memory=False)
assert_is_not_none(world_food_data)
world_food_data.head()
print("Total {} observations on {} features".format(world_food_data.shape[0],world_food_data.shape[1]))
cols_to_keep=["product_name","packaging","main_category","nutrition_grade_fr",
              "nutrition-score-fr_100g","fat_100g","carbohydrates_100g","proteins_100g",
               "additives_n","ingredients_from_palm_oil_n","first_packaging_code_geo"]
world_food_data=world_food_data[cols_to_keep]
world_food_data=world_food_data.rename(columns={"nutrition-score-fr_100g":"nutrition_score",
                                                "fat_100g":"fat_g",
                                               "carbohydrates_100g":"carbohydrates_g",
                                               "proteins_100g":"proteins_g"})
assert_equal(world_food_data.shape[1],11)
world_food_data.head()
len(world_food_data[world_food_data.packaging.isnull()])
len(world_food_data[world_food_data.first_packaging_code_geo.isnull()])
world_food_data.additives_n.unique()
world_food_data.ingredients_from_palm_oil_n.unique()
len(world_food_data[world_food_data.ingredients_from_palm_oil_n==2])
most_common_coords=world_food_data.first_packaging_code_geo.value_counts().index[0]
most_common_packaging=world_food_data.packaging.value_counts().index[0]
mean_additives=world_food_data.additives_n.mean()

world_food_data.additives_n.loc[world_food_data.additives_n.isnull()]=mean_additives
world_food_data.ingredients_from_palm_oil_n.loc[world_food_data.ingredients_from_palm_oil_n.isnull()]=0
world_food_data.first_packaging_code_geo.loc[world_food_data.first_packaging_code_geo.isnull()]=most_common_coords
world_food_data.packaging.loc[world_food_data.packaging.isnull()]=most_common_packaging

world_food_data=world_food_data.dropna()
print("Total {} observations on {} features".format(world_food_data.shape[0],world_food_data.shape[1]))
assert_is_not_none(most_common_coords)
assert_is_not_none(most_common_packaging)
assert_is_not_none(mean_additives)
assert_false(world_food_data.any().isnull().any())
assert_equal(world_food_data.shape,(71091,11))
world_food_data.dtypes
world_food_data.nutrition_score.unique()
world_food_data.additives_n.unique()
world_food_data.ingredients_from_palm_oil_n.unique()
world_food_data.head()
world_food_data["main_category"]=world_food_data["main_category"].map(lambda x: str(x)[3:])
world_food_data[["additives_n","ingredients_from_palm_oil_n"]]=world_food_data[["additives_n","ingredients_from_palm_oil_n"]].astype(int)
world_food_data[["fp_lat","fp_lon"]]=world_food_data["first_packaging_code_geo"].str.split(",", 1, expand=True)
world_food_data.fp_lat=round(world_food_data.fp_lat.astype(float),2)
world_food_data.fp_lon=round(world_food_data.fp_lon.astype(float),2)
world_food_data=world_food_data.drop(columns="first_packaging_code_geo")

world_food_data.nutrition_score=world_food_data.nutrition_score.astype(int)

world_food_data=world_food_data.reset_index(drop=True)
assert_equal(world_food_data.fp_lat.dtype,float)
assert_equal(world_food_data.fp_lon.dtype,float)
assert_equal(world_food_data.nutrition_score.dtype,int)
assert_equal(world_food_data.shape[1],12)
world_food_data.dtypes
world_food_data.head()
world_food_data["contains_additives"]=pd.Series(np.where(world_food_data.additives_n>0,1,0)).astype(int)
world_food_data.packaging=world_food_data.packaging.str.lower()
assert_less(world_food_data.contains_additives.any(),2)
assert_greater_equal(world_food_data.contains_additives.any(),0)
assert_equal(world_food_data.shape[1],13)
world_food_data.head()
world_food_data["ingredients_from_palm_oil_n"].unique()
len(world_food_data[world_food_data.ingredients_from_palm_oil_n==2])
world_food_data["ingredients_from_palm_oil_n"].loc[world_food_data["ingredients_from_palm_oil_n"]==2]=1
assert_greater_equal(world_food_data.ingredients_from_palm_oil_n.any(),0)
assert_less_equal(world_food_data.ingredients_from_palm_oil_n.any(),1)
starbucks_data=pd.read_csv("../input/starbucks-menu/starbucks_drinkMenu_expanded.csv")
assert_is_not_none(starbucks_data)
starbucks_data.head()
starbucks_data.columns=starbucks_data.columns.str.replace(")","")
starbucks_data.columns=starbucks_data.columns.str.replace(" ","")
starbucks_data.columns=starbucks_data.columns.str.replace("(","_")
starbucks_data.columns=starbucks_data.columns.str.lower()
cols_to_keep=["beverage_category", "beverage","beverage_prep","calories","totalfat_g","totalcarbohydrates_g",
               "protein_g"]
starbucks_data=starbucks_data[cols_to_keep]
starbucks_data=starbucks_data.rename(columns={"totalfat_g":"fat_g",
                                                "totalcarbohydrates_g":"carbohydrates_g",
                                              "protein_g":"proteins_g",
                                              "beverage":"product_name"
                                               })
starbucks_data.shape
assert_equal(starbucks_data.shape,(242,7))
starbucks_data.head()
starbucks_data.dtypes
starbucks_data.carbohydrates_g=starbucks_data.carbohydrates_g.astype(float)
assert_equal(starbucks_data.carbohydrates_g.dtype,float)
starbucks_data.fat_g.unique()
starbucks_data_mistake=starbucks_data.fat_g.loc[starbucks_data.fat_g=="3 2"]
starbucks_data.fat_g=starbucks_data.fat_g.replace(starbucks_data_mistake,np.nan)
starbucks_data.fat_g=starbucks_data.fat_g.astype(float)
assert_equal(starbucks_data.fat_g.dtype,float)
starbucks_data.dtypes
starbucks_data.head()
mcd_menu_data=pd.read_csv("../input/nutrition-facts/menu.csv")
assert_is_not_none(mcd_menu_data)
mcd_menu_data.columns=mcd_menu_data.columns.str.replace(")","")
mcd_menu_data.columns=mcd_menu_data.columns.str.replace(" ","")
mcd_menu_data.columns=mcd_menu_data.columns.str.replace("(","_")
mcd_menu_data.columns=mcd_menu_data.columns.str.lower()
cols_to_keep=["category","item","calories","totalfat","carbohydrates","protein"]
mcd_menu_data=mcd_menu_data[cols_to_keep]
mcd_menu_data=mcd_menu_data.rename(columns={"totalfat":"fat_g",
                                            "item":"product_name",
                                           "carbohydrates":"carbohydrates_g",
                                           "protein":"proteins_g"
                                           })
mcd_menu_data.shape
assert_equal(mcd_menu_data.shape,(260,6))
mcd_menu_data.head()
mcd_menu_data.dtypes
mcd_menu_data.carbohydrates_g=mcd_menu_data.carbohydrates_g.astype(float)
mcd_menu_data.proteins_g=mcd_menu_data.proteins_g.astype(float)
assert_equal(mcd_menu_data.carbohydrates_g.dtype,float)
assert_equal(mcd_menu_data.proteins_g.dtype,float)
mcd_menu_data.head()
mcd_menu_data.proteins_g.unique()
mcd_menu_data.carbohydrates_g.unique()
mcd_menu_data.to_csv("mcd_menu_scrubbed.csv",index=False)
starbucks_data.to_csv("star_menu_scrubbed.csv",index=False)
world_food_data.to_csv("world_food_scrubbed.csv",index=False)