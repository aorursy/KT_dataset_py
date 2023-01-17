import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from nose.tools import *

from mpl_toolkits.basemap import Basemap
world_food_data=pd.read_csv("../input/openfoodfactsclean/world_food_scrubbed.csv")
starbucks_data=pd.read_csv("../input/openfoodfactsclean/star_menu_scrubbed.csv")
mcd_menu_data=pd.read_csv("../input/openfoodfactsclean/mcd_menu_scrubbed.csv")
assert_is_not_none(world_food_data)
assert_is_not_none(starbucks_data)
assert_is_not_none(mcd_menu_data)
world_food_data.head()
starbucks_data.head()
mcd_menu_data.head()
print("Dataset shapes for exploration: MCD({},{}), Starbucks({},{}), World food({},{})".format(mcd_menu_data.shape[0],mcd_menu_data.shape[1],
                                                                              starbucks_data.shape[0],starbucks_data.shape[1],
                                                                              world_food_data.shape[0],world_food_data.shape[1]))
assert_equal(world_food_data.shape,(71091,13))
assert_equal(starbucks_data.shape,(242,7))
assert_equal(mcd_menu_data.shape,(260,6))
world_food_data["packaging"].groupby(world_food_data["packaging"]).count().sort_values(ascending=False)
products_plastic=world_food_data.loc[world_food_data["packaging"].str.contains("plastique")]
products_cardboard=world_food_data.loc[world_food_data["packaging"].str.contains("carton")]
products_canned=world_food_data.loc[world_food_data["packaging"].str.contains("conserve")]

num_canned,num_cardboard,num_plastic=(products_canned["packaging"].count(),
                                      products_cardboard["packaging"].count(),
                                      products_plastic["packaging"].count())
assert_is_not_none(products_plastic)
assert_is_not_none(products_cardboard)
assert_is_not_none(products_canned)
assert_greater(num_canned,0)
assert_greater(num_cardboard,0)
assert_greater(num_plastic,0)
plt.title("Distribution of different packaging")
plt.bar(range(3), [num_canned,num_cardboard,num_plastic])
plt.xticks(range(3), ["Canned", "Cardboard", "Plastic"])
plt.ylabel("Packaging Count")
plt.show()
plt.title("Distribution of nutrition scores by different packaging")
plt.ylabel("Packaging count")
plt.xlabel("Nutrition score")
plt.hist(products_plastic["nutrition_score"],bins=20,alpha=0.7)
plt.hist(products_cardboard["nutrition_score"],bins=20,alpha=0.7)
plt.hist(products_canned["nutrition_score"],bins=20,alpha=0.7)
plt.legend(["Plastic", "Cardboard", "Canned"])
plt.show()
plt.title("Distribution of additive count in products")
plt.xlabel("Additive count")
plt.ylabel("Additive count distribution")
plt.hist(world_food_data["additives_n"])
plt.show()
world_food_data["additives_n"].unique()
products_with_additives=world_food_data["contains_additives"].groupby(world_food_data["contains_additives"]).count()
products_with_additives
products_with_additives.index=["don't contain additives","contain additives"]
assert_equal(len(products_with_additives),2)
def plot_pie_on_grouped_data(grouped_data,title,explode):
    plt.gca().set_aspect("equal")
    plt.pie(grouped_data,labels=grouped_data.index, autopct = "%.2f%%",explode=explode,radius=1)
    plt.title(title)
    plt.show()
plot_pie_on_grouped_data(products_with_additives,"Percentage of french products containing additives",(0,0.1))
additives_by_grade=world_food_data["additives_n"].groupby(world_food_data["nutrition_grade_fr"])
assert_equal(len(additives_by_grade),5)
for additive, grade in additives_by_grade:
    plt.hist(grade, label = "Grade {}".format(additive), alpha = 0.5)
plt.title("Distribution of additive count by nutrition grade")
plt.xlabel("Additive count")
plt.ylabel("Additive count distribution")
plt.legend()
plt.show()
palm_oil_group=world_food_data["ingredients_from_palm_oil_n"].groupby(world_food_data
                                                                         ["ingredients_from_palm_oil_n"]).count()
palm_oil_group
palm_oil_group.index=["palm oil absent","palm oil present"]
assert_equal(palm_oil_group.values.tolist(),[67055,4036])
assert_equal(palm_oil_group.index.tolist(),["palm oil absent","palm oil present"])
palm_oil_group
plot_pie_on_grouped_data(palm_oil_group,"French products with and without palm oil ingredients",(0,0.1))
num_products_by_category=world_food_data.main_category.groupby(world_food_data.main_category).count().sort_values(ascending=False).nlargest(10)
assert_equal(len(num_products_by_category),10)
def plot_barh_on_grouped_data(grouped_data,title,y_label,fig_size):
    plt.figure(figsize = fig_size)
    plt.title(title)
    plt.ylabel(y_label)
    plt.barh(range(len(grouped_data)), grouped_data)
    plt.yticks(list(range(len(grouped_data))), grouped_data.index)
    plt.show()
plot_barh_on_grouped_data(num_products_by_category,"French product categories with the highest count","",(10,6))
grades_by_category=world_food_data.main_category.groupby(world_food_data.nutrition_grade_fr).count()
assert_equal(len(grades_by_category),5)
grades_by_category
plot_barh_on_grouped_data(grades_by_category,"Nutrition grade distributions for the french products","Nutrition grade",(10,6))
french_beverages=world_food_data[["product_name","fat_g","carbohydrates_g","proteins_g"]].loc[world_food_data["main_category"]=="beverages"]
starbucks_beverages=starbucks_data[["product_name","beverage_prep","fat_g","carbohydrates_g","proteins_g"]]
assert_equal(french_beverages.shape[1],4)
assert_equal(starbucks_beverages.shape[1],5)
starbucks_beverages.corr()
french_beverages.corr()
print("Number of french beverages:{} , Number of Starbucks beverages:{}".format(french_beverages.shape[0],
                                                                               starbucks_beverages.shape[0]))
print("Number of unique french beverages:{}, Number of unique Starbucks beverages:{}".format(len(french_beverages.product_name.unique()),
                                                                                             len(starbucks_beverages.product_name.unique())))
def extract_mean_total_nutrients(larger_df,smaller_df,num_iterations):
    total_sum=0
    larger_df_copy=larger_df.copy()
    smaller_df_copy=smaller_df.copy()
    
    #Check if the larger dataframe is actually given as the second parameter and switch the dataframes
    if larger_df.shape[0] < smaller_df.shape[0]:
        larger_df=smaller_df_copy
        smaller_df=larger_df_copy
        
    for i in range(num_iterations):
        total_nutrients_larger = round(larger_df.carbohydrates_g.sample(len(smaller_df)).sum() + larger_df.proteins_g.sample(len(smaller_df)).sum() + larger_df.fat_g.sample(len(smaller_df)).sum())
        total_nutrients_smaller = round(smaller_df.carbohydrates_g.sum() + smaller_df.proteins_g.sum() + smaller_df.fat_g.sum())
        print("Sample ",i+1)
        print("Total sampled nutrients (Larger dataframe):{} , Total nutrients (Smaller dataframe):{}".format(total_nutrients_larger,
                                                                                        total_nutrients_smaller))
        sample_per=total_nutrients_larger/total_nutrients_smaller*100
        print("Total % of nutrients in iteration for the larger dataframe:{:.2f}".format(sample_per))
        total_sum+=sample_per
    
    total_mean=total_sum/num_iterations
    print("\nMean total % of nutrients for the larger dataframe across all iterations:{:.2f}".format(total_mean))
    return total_mean
mean_result=extract_mean_total_nutrients(french_beverages,starbucks_beverages,10)
assert_greater(mean_result,5)
assert_less(mean_result,20)
def get_max_product_values_by_category(dataframe,category):
    group_result=category.groupby(dataframe.product_name).max().sort_values(ascending=False).nlargest(10)
    return group_result
carb_heavy_french_beverages=get_max_product_values_by_category(french_beverages,french_beverages.carbohydrates_g)
carb_heavy_starbucks_beverages=get_max_product_values_by_category(starbucks_beverages,starbucks_beverages.carbohydrates_g)
carb_heavy_french_beverages
filter_out_list = ['Sirop', 'SIROP', 'sirop', 'Agaven', 'agaven', 'AGAVEN',
                                                        'Dessert', 'DESSERT', 'dessert', 'Bonbons']
pattern='|'.join(filter_out_list)
french_beverages = french_beverages[~french_beverages.product_name.str.contains(pattern)]
carb_heavy_french_beverages=get_max_product_values_by_category(french_beverages,french_beverages.carbohydrates_g)
carb_heavy_french_beverages
carb_heavy_starbucks_beverages
assert_equal(len(carb_heavy_french_beverages),10)
assert_equal(len(carb_heavy_starbucks_beverages),10)
assert_equal(carb_heavy_french_beverages.values[0],99)
assert_equal(carb_heavy_starbucks_beverages.values[0],340)
plot_barh_on_grouped_data(carb_heavy_french_beverages,"French beverages that contain the highest amount of carbohydrates","",(10,6))
plot_barh_on_grouped_data(carb_heavy_starbucks_beverages,"Starbucks beverages that contain the highest amount of carbohydrates","",(10,6))
print(round(carb_heavy_starbucks_beverages[0]/carb_heavy_french_beverages[4]))
french_meat_data=world_food_data[["product_name","fat_g","carbohydrates_g","proteins_g"]].loc[world_food_data["main_category"]=="meats"]
words_to_search=["Sausage","Bacon","Chicken","Steak"]
pattern='|'.join(words_to_search)
mcd_meat_data=mcd_menu_data[["product_name","fat_g","carbohydrates_g","proteins_g"]].loc[(mcd_menu_data["category"]=='Beef & Pork') | 
                                                                                         (mcd_menu_data["category"]=='Chicken & Fish')]
mcd_meat_data=mcd_meat_data.append(mcd_menu_data[["product_name","fat_g","carbohydrates_g","proteins_g"]].
                                   loc[mcd_menu_data.product_name.str.contains(pattern)])
mcd_meat_data=mcd_meat_data[~mcd_meat_data["product_name"].isin(["Fish"])]
mcd_meat_data.shape
print("Number of french meat products:{} , Number of McDonalds meat products:{}".format(french_meat_data.shape[0],
                                                                               mcd_meat_data.shape[0]))
assert_equal(french_meat_data.shape,(3790,4))
assert_equal(mcd_meat_data.shape,(111,4))
mcd_meat_data.corr()
french_meat_data.corr()
mean_result = extract_mean_total_nutrients(french_meat_data,mcd_meat_data,10)
assert_greater(mean_result,30)
assert_less(mean_result,45)
fat_heavy_french_meat_products=get_max_product_values_by_category(french_meat_data,french_meat_data.fat_g)
fat_heavy_mcd_meat_products=get_max_product_values_by_category(mcd_meat_data,mcd_meat_data.fat_g)
fat_heavy_french_meat_products
fat_heavy_mcd_meat_products
assert_equal(len(fat_heavy_french_meat_products),10)
assert_equal(len(fat_heavy_mcd_meat_products),10)
assert_equal(fat_heavy_french_meat_products.values[0],73)
assert_equal(fat_heavy_mcd_meat_products.values[0],118)
plot_barh_on_grouped_data(fat_heavy_french_meat_products,"French meat products with the highest amount of fat","",(10,6))
plot_barh_on_grouped_data(fat_heavy_mcd_meat_products,"McDonalds products with the highest amount of fat","",(10,6))
french_meat_data["grade"]=world_food_data["nutrition_grade_fr"]
assert_equal(french_meat_data.shape[1],5)
french_meat_data.head()
def group_by_grade_and_make_hypotheses(dataframe,category):
    group_result=category.groupby(dataframe.grade)
    print("Category mean by:{}".format(group_result.mean()))
    category_grade_a=group_result.get_group("a")
    category_grade_b=group_result.get_group("b")
    category_grade_d=group_result.get_group("d")
    hyp_ab = ttest_ind(category_grade_a,category_grade_b,equal_var=False)
    hyp_bd = ttest_ind(category_grade_b,category_grade_d,equal_var=False)
    hyp_ad = ttest_ind(category_grade_a,category_grade_d,equal_var=False)
    print("A->B:{}".format(hyp_ab.pvalue))
    print("B->D:{}".format(hyp_bd.pvalue))
    print("A->D:{}".format(hyp_ad.pvalue))
    if hyp_ab.pvalue <= 0.01 and hyp_bd.pvalue <= 0.01 and hyp_ad.pvalue <= 0.01:
        print("The differences in grades are significant. Reject H0.")
    else:
        print("There's not enough evidence to reject H0. Don't accept or reject anything else.")
    return (hyp_ab,hyp_bd,hyp_ad)
(test_fat_result_ab,test_fat_result_bd,test_fat_result_ad)=group_by_grade_and_make_hypotheses(french_meat_data,french_meat_data.fat_g)
assert_is_not_none((test_fat_result_ab,test_fat_result_bd,test_fat_result_ad))
(test_carb_result_ab,test_carb_result_bd,test_carb_result_ad)=group_by_grade_and_make_hypotheses(french_meat_data,french_meat_data.carbohydrates_g)
assert_is_not_none((test_carb_result_ab,test_carb_result_bd,test_carb_result_ad))
(test_prot_result_ab,test_prot_result_bd,test_prot_result_ad)=group_by_grade_and_make_hypotheses(french_meat_data,french_meat_data.proteins_g)
assert_is_not_none((test_prot_result_ab,test_prot_result_bd,test_prot_result_ad))
def draw_map_of_french_products(df_latitude,df_longitude,lat_lower_left,lon_lower_left,lat_upper_right,lon_upper_right,title):
    plt.figure(figsize = (12, 10))
    m = Basemap(projection = "merc", llcrnrlat = lat_lower_left, llcrnrlon = lon_lower_left, urcrnrlat = lat_upper_right, urcrnrlon = lon_upper_right)
    x, y = m(df_longitude.tolist(),df_latitude.tolist())
    m.plot(x,y,'o',markersize=1,color='red')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color = "lightgreen", lake_color = "aqua")
    m.drawmapboundary(fill_color = "aqua")
    plt.title(title)
    plt.show()
draw_map_of_french_products(world_food_data.fp_lat,world_food_data.fp_lon,-73,-180,80,180,"First packaging of the French products")
draw_map_of_french_products(world_food_data.fp_lat,world_food_data.fp_lon,20,-20,52,20,"First packaging of the French products zoomed")