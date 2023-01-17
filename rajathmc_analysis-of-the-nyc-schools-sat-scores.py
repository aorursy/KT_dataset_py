import pandas
import numpy as np

files = ["../input/ap_2010.csv", "../input/class_size.csv", "../input/demographics.csv", "../input/graduation.csv", "../input/hs_directory.csv", "../input/math_test_results.csv", "../input/sat_results.csv"]

data = {}
for f in files:
    d = pandas.read_csv("{0}".format(f))
    data[f.replace(".csv", "")] = d
for k,v in data.items():
    print("\n" + k + "\n")
    print(v.head())
data["../input/demographics"]["DBN"].head()
data["../input/class_size"].head()
data["../input/class_size"]["DBN"] = data["../input/class_size"].apply(lambda x: "{0:02d}{1}".format(x["CSD"], x["SCHOOL CODE"]), axis=1)
data["../input/hs_directory"]["DBN"] = data["../input/hs_directory"]["dbn"]
survey1 = pandas.read_csv("../input/survey_all.txt", delimiter="\t", encoding='windows-1252')
survey2 = pandas.read_csv("../input/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey1["d75"] = False
survey2["d75"] = True
survey = pandas.concat([survey1, survey2], axis=0)
survey.head()
survey["DBN"] = survey["dbn"]
survey_fields = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", "eng_t_10", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11", "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11",]
survey = survey.loc[:,survey_fields]
data["survey"] = survey
survey.shape
data["../input/class_size"].head()
data["../input/sat_results"].head()
class_size = data["../input/class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]
class_size = class_size.groupby("DBN").agg(np.mean)
class_size.reset_index(inplace=True)
data["../input/class_size"] = class_size
demographics = data["../input/demographics"]
demographics = demographics[demographics["schoolyear"] == 20112012]
data["../input/demographics"] = demographics
data["../input/math_test_results"] = data["../input/math_test_results"][data["../input/math_test_results"]["Year"] == 2011]
data["../input/math_test_results"] = data["../input/math_test_results"][data["../input/math_test_results"]["Grade"] == '8']
data["../input/graduation"] = data["../input/graduation"][data["../input/graduation"]["Cohort"] == "2006"]
data["../input/graduation"] = data["../input/graduation"][data["../input/graduation"]["Demographic"] == "Total Cohort"]
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["../input/sat_results"][c] = data["../input/sat_results"][c].convert_objects(convert_numeric=True)

data['../input/sat_results']['sat_score'] = data['../input/sat_results'][cols[0]] + data['../input/sat_results'][cols[1]] + data['../input/sat_results'][cols[2]]
data["../input/hs_directory"]['lat'] = data["../input/hs_directory"]['Location 1'].apply(lambda x: x.split("\n")[-1].replace("(", "").replace(")", "").split(", ")[0])
data["../input/hs_directory"]['lon'] = data["../input/hs_directory"]['Location 1'].apply(lambda x: x.split("\n")[-1].replace("(", "").replace(")", "").split(", ")[1])

for c in ['lat', 'lon']:
    data["../input/hs_directory"][c] = data["../input/hs_directory"][c].convert_objects(convert_numeric=True)
for k,v in data.items():
    print(k)
    print(v.head())
flat_data_names = [k for k,v in data.items()]
flat_data = [data[k] for k in flat_data_names]
full = flat_data[0]
for i, f in enumerate(flat_data[1:]):
    name = flat_data_names[i+1]
    print(name)
    print(len(f["DBN"]) - len(f["DBN"].unique()))
    join_type = "inner"
    if name in ["../input/sat_results", "../input/ap_2010", "../input/graduation"]:
        join_type = "outer"
    if name not in ["../input/math_test_results"]:
        full = full.merge(f, on="DBN", how=join_type)

full.shape
cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    full[col] = full[col].convert_objects(convert_numeric=True)

full[cols] = full[cols].fillna(value=0)
full["school_dist"] = full["DBN"].apply(lambda x: x[:2])
full = full.fillna(full.mean())
full.corr()['sat_score']
import folium
from folium import plugins
from folium.plugins import MarkerCluster

schools_map = folium.Map(location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
marker_cluster = MarkerCluster().add_to(schools_map)
for name, row in full.iterrows():
    folium.Marker([row["lat"], row["lon"]], popup="{0}: {1}".format(row["DBN"], row["school_name"])).add_to(marker_cluster)
schools_map.save('schools.html')
schools_map
schools_heatmap = folium.Map(location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
schools_heatmap.add_children(plugins.HeatMap([[row["lat"], row["lon"]] for name, row in full.iterrows()]))
schools_heatmap.save("heatmap.html")
schools_heatmap
district_data = full.groupby("school_dist").agg(np.mean)
district_data.reset_index(inplace=True)
district_data["school_dist"] = district_data["school_dist"].apply(lambda x: str(int(x)))
def show_district_map(col):
    geo_path = '../input/schools/districts.geojson'
    d = district_data
    map = folium.Map(location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
    map.geo_json(geo_path=geo_path, data=d, columns=['school_dist', col], key_on='feature.properties.school_dist', fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2)
    map.create_map(path="districts.html")
    return districts
%matplotlib inline

full.plot.scatter(x='total_enrollment', y='sat_score')
full[(full["total_enrollment"] < 1000) & (full["sat_score"] < 1000)]["School Name"]
full.plot.scatter(x='ell_percent', y='sat_score')
full.corr()["sat_score"][["rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_tot_11", "com_tot_11", "aca_tot_11", "eng_tot_11"]].plot.bar()
full.corr()["sat_score"][["white_per", "asian_per", "black_per", "hispanic_per"]].plot.bar()
full.corr()["sat_score"][["male_per", "female_per"]].plot.bar()
full.plot.scatter(x='female_per', y='sat_score')
full[(full["female_per"] > 65) & (full["sat_score"] > 1400)]["School Name"]
full["ap_avg"] = full["AP Test Takers "] / full["total_enrollment"]

full.plot.scatter(x='ap_avg', y='sat_score')
full[(full["ap_avg"] > .3) & (full["sat_score"] > 1700)]["School Name"]
