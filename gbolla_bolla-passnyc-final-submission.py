import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
import plotly.plotly as py

input_data_dir = os.getcwd().replace('/working', '/input')
final_data_set = pd.read_csv("{}/passnyc-created-data/step5_full_processed_test_dataset.csv".format(input_data_dir))
nyc_img=mpimg.imread('{}/passnyc-created-data/cropped_neighbourhoods_new_york_city_map.png'.format(input_data_dir))

pos_test_set_df = final_data_set[final_data_set["positive_diff_num_testtakers"] == True]
neg_test_set_df = final_data_set[final_data_set["positive_diff_num_testtakers"] == True]

final_data_set.plot(kind="scatter", 
    x="Longitude", 
    y="Latitude", 
    c=final_data_set["pct_poverty_2017_val"].astype(float), 
    s=final_data_set["diff_num_testtakers_all"].astype(float) * 25, 
    cmap=plt.get_cmap("coolwarm"), 
    title='Additional Expected Number of Testtakers (Color denotes % Below Poverty Level)',
    figsize=(16.162,16),
    alpha=0.5)
plt.imshow(nyc_img, 
           extent=[float(final_data_set["Longitude"].min() - .01) , float(final_data_set["Longitude"].max() + .01), float(final_data_set["Latitude"].min() - .01), float(final_data_set["Latitude"].max() + .01)], 
           alpha = 0.25)
plt.show()
pos_data = [
    {
        'x': pos_test_set_df["Longitude"],
        'y': pos_test_set_df["Latitude"],
        'text': pos_test_set_df["addtl_testtakers_label"],
        'mode': 'markers',
        'marker': {
            'color': pos_test_set_df["pct_poverty_2017_val"].astype(float),
            'size': pos_test_set_df["diff_num_testtakers_all"].astype(float) * 1.2,
            'showscale': True,
            'colorscale':'RdBu',
            'opacity':0.5
        }
    }
]

layout= go.Layout(
    autosize=False,
    width=900,
    height=750,
    title= 'Additional Expected Number of Testtakers (Color denotes % Below Poverty Level)',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    ))
fig=go.Figure(data=pos_data,layout=layout)
plotly.offline.iplot(fig, filename='scatter_hover_labels')
import os
import sys
import pandas as pd
import warnings
import requests
from bs4 import BeautifulSoup
import numpy as np
import re
from selenium import webdriver
import time
import math

def nyt_web_scrape(data_dir):

    print("\n*****scraping NYT article with SHSAT stats")
    nyt_schools_url = "https://www.nytimes.com/interactive/2018/06/29/nyregion/nyc-high-schools-middle-schools-shsat-students.html"
    csv_drop_path = "{}/step1_nyt_shsat_article_data.csv".format(data_dir)
    
    schools_html = requests.get(nyt_schools_url, verify=False).content
    schools_html = schools_html.decode("utf-8")
    schools_soup = BeautifulSoup(re.sub("<!--|-->","", schools_html), "lxml") 

    schools_table = schools_soup.find(class_="g-schools-table-container").table.tbody
    school_rows = schools_table.findAll('tr')

    nyt_article_cols = [
        "school_name",
        "dbn",
        "num_testtakers",
        "num_offered",
        "pct_8th_graders_offered",
        "pct_black_hispanic"
    ]
    output_df = pd.DataFrame(columns = nyt_article_cols)
    
    for school in school_rows:
        
        school_dict = {}        
        school_name = school['data-name']
        dbn = school['data-dbn']
        school_dict['school_name'] = school_name
        school_dict['dbn'] = dbn

        school_data = school.findAll('td')

        for td in school_data:

            school_stat = td['class']

            if "g-testers" in school_stat :
                school_dict['num_testtakers'] = td.string
            elif "g-offers" in school_stat:
                school_dict['num_offered'] = td.string
            elif "g-offers-per-student" in school_stat:
                school_dict['pct_8th_graders_offered'] = td.string
            elif "g-pct" in school_stat:
                school_dict['pct_black_hispanic'] = td.string

        output_df = output_df.append(school_dict, ignore_index = True)

    
    merged_df = merge_w_explorer_data(output_df, data_dir)
    print("-- dropping NYT article CSV to {}".format(csv_drop_path))
    merged_df.to_csv(csv_drop_path, index = False)

    return merged_df

def merge_w_explorer_data(nyt_df, data_dir):

    school_explorer_path = data_dir.replace('external', 'raw')
    print(school_explorer_path)
    school_explorer_df = pd.read_csv("{}/2016 School Explorer.csv".format(school_explorer_path))
    
    nyt_df[["num_testtakers", "num_offered"]] = nyt_df[["num_testtakers", "num_offered"]].replace(to_replace="—",value=0)
    nyt_df["pct_8th_graders_offered"] = nyt_df["pct_8th_graders_offered"].replace(to_replace="—",value="0%")
    
    school_explorer_df_cols_to_keep = [
        'School Name',
        'SED Code',
        'Location Code',
        'District',
        'Latitude',
        'Longitude',
        'Address (Full)',
        'City',
        'Zip',
        'Grades',
        'Grade Low',
        'Grade High',
        'Community School?',
        'Economic Need Index',
        'School Income Estimate',
        'Percent ELL',
        'Percent Asian',
        'Percent Black',
        'Percent Hispanic',
        'Percent Black / Hispanic',
        'Percent White',
        'Student Attendance Rate',
        'Percent of Students Chronically Absent',
        'Rigorous Instruction %',
        'Rigorous Instruction Rating',
        'Collaborative Teachers %',
        'Collaborative Teachers Rating',
        'Supportive Environment %',
        'Supportive Environment Rating',
        'Effective School Leadership %',
        'Effective School Leadership Rating',
        'Strong Family-Community Ties %',
        'Strong Family-Community Ties Rating',
        'Trust %',
        'Trust Rating',
        'Student Achievement Rating',
        'Average ELA Proficiency',
        'Average Math Proficiency'
    ]

    trimmed_school_explorer_df = school_explorer_df[school_explorer_df_cols_to_keep]
    
    merged_df = pd.merge(nyt_df, trimmed_school_explorer_df, left_on="dbn", right_on="Location Code", how="outer")
    
    # combine school_name column from nyt and school explorer data
    merged_df['school_name'] = np.where(merged_df['school_name'].isnull(), merged_df['School Name'], merged_df['school_name'])
    merged_df['dbn'] = np.where(merged_df['dbn'].isnull(), merged_df['Location Code'], merged_df['dbn'])

    merged_df = merged_df.drop(['School Name', 'Location Code'], axis=1)

    return merged_df

wait_time_after_click = 0.5
wait_time_after_exception = 30

def dept_of_ed_web_scrape(school_df, data_dir, start_idx = 0 , debug_flg = False):

    print("\n*****scraping NYC DOE")
    if debug_flg == False:
        output_df = pd.DataFrame()
    else:
        output_df = pd.read_csv("{data_dir}/step2_in_flight_doe_nyt_data.csv".format(**locals()))
        output_df = output_df.drop_duplicates()

    DoE_base_url = "https://tools.nycenet.edu/guide/{year}/#dbn={dbn}&report_type=EMS"

    for idx, school in school_df[int(start_idx):].iterrows():

        scrape_school_flg = get_scrape_flg(school, DoE_base_url)

        if scrape_school_flg == False: 
            continue

        row_dict = get_school_info(school, DoE_base_url, idx)
        output_df = output_df.append(row_dict, ignore_index=True)
        output_df.to_csv("{data_dir}/step2_in_flight_doe_nyt_data.csv".format(**locals()),index=False)
    output_df.to_csv("{data_dir}/step2_final_doe_nyt_data.csv".format(**locals()),index=False)
    
def get_school_info(school, DoE_base_url, idx):
    
    years_to_scrape = [2017]
    school_row_dict = school

    school_name = school['school_name']
    dbn = school['dbn']

    browser = webdriver.Chrome()

    # if there's an issue scraping data from a particular school, 
    # you can just pass in the index and process will continue from there
    print("school: {} / index: {}".format(school_name, idx))
    for year in years_to_scrape:
        school_url = DoE_base_url.format(year=year, dbn=dbn)
        print("--{}".format(school_url))
        
        try:
            browser.get(school_url)
            browser.find_element_by_class_name('introjs-skipbutton').click()
        except:
            print("---in except, waiting {} seconds and retrying".format(wait_time_after_exception))
            time.sleep(wait_time_after_exception)
            browser.get(school_url)
            time.sleep(wait_time_after_exception)
            browser.find_element_by_class_name('introjs-skipbutton').click() 
            
        time.sleep(wait_time_after_click)

        school_row_dict = get_student_achievement_stats(browser, school_row_dict, year)
        school_row_dict = get_student_characteristic_stats(browser, school_row_dict, year)
        
    browser.quit()
    return school_row_dict

def uncollapse_all(browser):
    collapsible_content = browser.find_elements_by_class_name('osp-collapsible-title')
    for x in range(0, len(collapsible_content)):
        if collapsible_content[x].is_displayed():
            collapsible_content[x].click()
            time.sleep(wait_time_after_click)

def get_student_characteristic_stats(browser, school_row_dict, year):
    
    browser.find_element_by_id('tab-stu-pop').click()
    time.sleep(wait_time_after_click)

    uncollapse_all(browser)

    student_characteristic_soup = BeautifulSoup(browser.page_source, "lxml") 

    enrollment_section = student_characteristic_soup.find(id="pop-eot")
    enrollment_content = enrollment_section.find(class_="osp-collapsible-content-wrapper")
    
    for enrollment_grade in enrollment_content.children:
        enrollment_data = enrollment_grade.find(class_="osp-collapsible-title")
        grade = enrollment_data.find(class_="name").string.split("Grade ")[-1]
        try:
            grade = int(grade)
        except ValueError:
            pass
        
        if grade not in [7, 8]:
            continue
        
        class_str = "yr-"
        regex = re.compile(".*({class_str}).*".format(**locals()))
        for child in enrollment_data.children:

            if any('yr-' in string for string in child['class']):
                idx = [i for i, s in enumerate(child['class']) if 'yr-' in s][0]
                class_yr_str = child['class'][idx]
                class_yr = int(class_yr_str.split('yr-')[-1])
                enrollment_yr = year - class_yr

                if enrollment_yr == year:
                    school_row_dict["grade_{}_{}_enrollment".format(grade, enrollment_yr)] = child.string

    addtl_resources_section = student_characteristic_soup.find(id="pop-hns")
    addtl_resources_content = addtl_resources_section.find(class_="cat-collapsibles")
    addtl_resources_name_dict = {
        "Students in Families Eligible for HRA Assistance" : "pct_hra_assistance",
        "Students in Families with Income Below Federal Poverty Level (Estimated)" : "pct_poverty",
        "Students in Temporary Housing" : "pct_temp_housing",
        "Economic Need Index" : "econ_need_index",
        "Students with Disabilities" : "pct_students_w_disabilities",
        "English Language Learners" : "pct_ell",
        "Avg 5th Grade ELA Rating" : "incoming_avg_5th_grade_ela_rating",
        "Avg 5th Grade Math Rating" : "incoming_avg_5th_grade_math_rating",
        "Math Level 1" : "incoming_math_level_1",
        "Math Level 2" : "incoming_math_level_2",
        "Math Level 3" : "incoming_math_level_3",
        "Math Level 4" : "incoming_math_level_4",
        "ELA Level 1" : "incoming_ela_level_1",
        "ELA Level 2" : "incoming_ela_level_2",
        "ELA Level 3" : "incoming_ela_level_3",
        "ELA Level 4" : "incoming_ela_level_4"
    }

    for addtl_resource_cat in addtl_resources_content.children:
        addtl_resource_cat_data = addtl_resource_cat.find(class_="osp-collapsible-content-wrapper")
    
        cat_section = addtl_resource_cat.find(class_="osp-collapsible-title")
        cat_section_name = cat_section.find(class_="name").div.string
        cat_section_val = cat_section.find(class_="val").string
        cat_section_dist_diff = cat_section.find(class_="dist").svg.text 
        cat_section_city_diff = cat_section.find(class_="city").svg.text 
        stat_col = addtl_resources_name_dict[cat_section_name]
        school_row_dict["{}_{}_val".format(stat_col, year)] = cat_section_val
        school_row_dict["{}_{}_dist_diff".format(stat_col, year)] = cat_section_dist_diff
        school_row_dict["{}_{}_city_diff".format(stat_col, year)] = cat_section_city_diff

        if cat_section_name == "Economic Need Index":
            for subcat in addtl_resource_cat_data.children:
                subcat_section = subcat.find(class_="osp-collapsible-title")
                subcat_section_name = subcat_section.find(class_="name").div.string
                stat_col = addtl_resources_name_dict[subcat_section_name]
                subcat_section_val = subcat_section.find(class_="val").string
                subcat_section_dist_diff = subcat_section.find(class_="dist").svg.text 
                subcat_section_city_diff = subcat_section.find(class_="city").svg.text 
                school_row_dict["{}_{}_val".format(stat_col, year)] = subcat_section_val
                school_row_dict["{}_{}_dist_diff".format(stat_col, year)] = subcat_section_dist_diff
                school_row_dict["{}_{}_city_diff".format(stat_col, year)] = subcat_section_city_diff


    incoming_proficiency = student_characteristic_soup.find(id="pop-ipl")
    ipl_content = incoming_proficiency.find(class_="cat-collapsibles")

    for ipl in ipl_content.children:
        ipl_data = ipl.find(class_="osp-collapsible-content-wrapper")
        ipl_section = ipl_data.find(class_="osp-collapsible-title")
        ipl_section_name = ipl_section.find(class_="name").div.string
        ipl_section_val = ipl_section.find(class_="val").string
        ipl_section_dist_diff = ipl_section.find(class_="dist").svg.text 
        ipl_section_city_diff = ipl_section.find(class_="city").svg.text 
        stat_col = addtl_resources_name_dict[ipl_section_name]
        school_row_dict["{}_{}_val".format(stat_col, year)] = ipl_section_val
        school_row_dict["{}_{}_dist_diff".format(stat_col, year)] = ipl_section_dist_diff
        school_row_dict["{}_{}_city_diff".format(stat_col, year)] = ipl_section_city_diff

        if ipl_section_name in ["Avg 5th Grade ELA Rating", "Avg 5th Grade Math Rating"]:
            if ipl_section_name == "Avg 5th Grade Math Rating":
                stat_col_prepend = "Math"
            elif ipl_section_name == "Avg 5th Grade ELA Rating":
                stat_col_prepend = "ELA"
            ipl_collapsible_children = ipl_data.find(class_="osp-collapsible-content-wrapper")
            for ipl_sub in ipl_collapsible_children.children:
                ipl_subsection = ipl_sub.find(class_="osp-collapsible-title")
                ipl_subsection_name = ipl_subsection.find(class_="name").div.string
                ipl_subsection_val = ipl_subsection.find(class_="val").string
                ipl_subsection_size = ipl_subsection.find(class_="n").string
                ipl_subsection_dist_diff = ipl_subsection.find(class_="dist").svg.text 
                ipl_subsection_city_diff = ipl_subsection.find(class_="city").svg.text
                stat_col = addtl_resources_name_dict["{} {}".format(stat_col_prepend, ipl_subsection_name)]
                school_row_dict["{}_{}_val".format(stat_col, year)] = ipl_subsection_val
                school_row_dict["{}_{}_n".format(stat_col, year)] = ipl_subsection_size
                school_row_dict["{}_{}_dist_diff".format(stat_col, year)] = ipl_subsection_dist_diff
                school_row_dict["{}_{}_city_diff".format(stat_col, year)] = ipl_subsection_city_diff

        time.sleep(wait_time_after_click)
    return school_row_dict

def get_student_achievement_stats(browser, school_row_dict, year):
    
    browser.find_element_by_id('tab-stu-achieve').click()
    time.sleep(wait_time_after_click)

    uncollapse_all(browser)
    
    sa_soup = BeautifulSoup(browser.page_source, "lxml") 

    sa_name_dict = {
        "White" : "white",
        "Hispanic" : "hispanic",
        "Asian / Pacific Islander" : "asian_pacific",
        "Black" : "black",
        "Multiracial" : "multiracial",
        "American Indian" : "amer_indian",
        "ELA - Average Student Proficiency" : "avg_ela_proficiency",
        "ELA - Percentage of Students at Level 3 or 4" : "pct_ela_level_3_or_4",
        "Math - Average Student Proficiency" : "avg_math_proficiency",
        "Math - Percentage of Students at Level 3 or 4" : "pct_math_level_3_or_4",
        "Percent of 8th Graders Earning HS Credit" : "pct_8th_graders_w_hs_credit"
    }

    sa_section = sa_soup.find(id="content-stu-achieve")
    sa_content = sa_section.find(class_="tab-content")

    sa_title = sa_content.find(class_="osp-collapsible-title")
    sa_score = sa_title.find(class_="score").string
    sa_score_dist_diff = sa_title.find(class_="dist").svg.text 
    sa_score_city_diff = sa_title.find(class_="city").svg.text

    sa_metric_collapsibles = sa_content.find(id="sa-metric-collapsibles")
    for sa_metric in sa_metric_collapsibles.children:
        sa_metric_content = sa_metric.find(class_="osp-collapsible-content-wrapper")

        for sa_stat in sa_metric_content.children:
            sa_title = sa_stat.find(class_="osp-collapsible-title") 
            try:
                stat_name = sa_title.find(class_="name").div.string
            except AttributeError:
                stat_name = sa_title.find(class_="name").string

            if stat_name in sa_name_dict.keys():
                stat_val = sa_title.find(class_="value").string
                sa_stat_comp_diff = sa_title.find(class_="comp").svg.text 
                sa_stat_city_diff = sa_title.find(class_="city").svg.text
                stat_col = sa_name_dict[stat_name]
                school_row_dict["{}_{}".format(stat_col, year)] = stat_val
                school_row_dict["{}_{}_comp_diff".format(stat_col, year)] = sa_stat_comp_diff
                school_row_dict["{}_{}_city_diff".format(stat_col, year)] = sa_stat_city_diff


    sa_addtl_info = sa_content.find(id="sa-add-info").find(class_="osp-collapsible-content-wrapper")
    
    sa_attendance_div = sa_addtl_info.find(id="sa-add-info-sg0-m0").find(class_="osp-collapsible-title")
    sa_attendance_name = sa_attendance_div.find(class_="name").div.string
    sa_attendance_val = sa_attendance_div.find(class_="value").string

    try:
        sa_attendance_dist_diff = sa_attendance_div.find(class_="dist").svg.text 
    except AttributeError:
        sa_attendance_dist_diff = sa_attendance_div.find(class_="comp").svg.text 

    sa_attendance_city_diff = sa_attendance_div.find(class_="city").svg.text 
    school_row_dict["sa_attendance_90plus_{}".format(year)] = sa_attendance_val
    school_row_dict["sa_attendance_90plus_{}_dist_diff".format(year)] = sa_attendance_dist_diff
    school_row_dict["sa_attendance_90plus_{}_city_diff".format(year)] = sa_attendance_city_diff


    sa_proficiency_scores_by_ethnicity = sa_addtl_info.find(id="sa-add-info-re-nonoverlap").find(class_="cat-demog-collapsibles")

    for sa_proficiency in sa_proficiency_scores_by_ethnicity.children:
        sa_proficiency_row = sa_proficiency.find(class_="osp-collapsible-title")
        ethnicity = sa_proficiency_row.find(class_="name").div.string
        if ethnicity == "Missing or Invalid Data":
            continue
        ethnicity_col_name = sa_name_dict[ethnicity]
        ethnicity_sample_size = sa_proficiency_row.find(class_="n").string
        incoming_ela = sa_proficiency_row.select('div.inc.ela')[0].string
        avg_ela = sa_proficiency_row.select('div.avg.ela')[0].string
        incoming_math = sa_proficiency_row.select('div.inc.mth')[0].string
        avg_math = sa_proficiency_row.select('div.avg.mth')[0].string 
        school_row_dict["{}_{}_num_students".format(ethnicity_col_name, year)] = ethnicity_sample_size
        school_row_dict["{}_{}_incoming_ela".format(ethnicity_col_name, year)] = incoming_ela
        school_row_dict["{}_{}_avg_ela".format(ethnicity_col_name, year)] = avg_ela
        school_row_dict["{}_{}_incoming_math".format(ethnicity_col_name, year)] = incoming_math
        school_row_dict["{}_{}_avg_math".format(ethnicity_col_name, year)] = avg_math    

    return school_row_dict

def get_scrape_flg(school_dict, DoE_base_url):

    
    if isinstance(school_dict['Grades'], float) and math.isnan(school_dict['Grades']):
        return False

    grade_list = school_dict['Grades'].split(",")

    for i, v in enumerate(grade_list): 
        try:
            grade_list[i] = int(v)
        except ValueError:
            grade_list[i] = v
        
    if 8 not in grade_list:
        return False

    else:
        try:
            scrape_flg = check_enrollment(school_dict, DoE_base_url)
        except Exception as e:
            if type(e).__name__ == "NoSuchElementException":
                url = DoE_base_url.format(year=2017, dbn=school_dict["dbn"])
                print("{} does not have data, returning False for scrape_flg".format(url))
                scrape_flg = False
                
        return scrape_flg

def check_enrollment(school_dict, DoE_base_url):
    
    dbn = school_dict['dbn']    
    school_url = DoE_base_url.format(year=2017, dbn=dbn)

    browser = webdriver.Chrome()
    browser.get(school_url)
    
    browser.find_element_by_class_name('introjs-skipbutton').click()
    time.sleep(wait_time_after_click)

    browser.find_element_by_id('tab-stu-pop').click()
    time.sleep(wait_time_after_click)

    enrollment_elem = browser.find_element_by_id('pop-eot')
    enrollment_elem.find_element_by_class_name('osp-collapsible-title').click()
    time.sleep(wait_time_after_click)

    enrollment_soup = BeautifulSoup(browser.page_source, "lxml") 
    enrollment_content = enrollment_soup.find(id="pop-eot").find(class_="osp-collapsible-content-wrapper")

    scrape_flg = False   

    for enrollment_grade in enrollment_content.children:
        enrollment_data = enrollment_grade.find(class_="osp-collapsible-title")
        grade = enrollment_data.find(class_="name").string.split("Grade ")[-1]

        try:
            grade = int(grade)

            if grade == 8:
                scrape_flg = True

        except ValueError:
            pass
    
    return scrape_flg
def web_scrape_controller(start_idx = 0, debug_flg = False):

    data_drop_dir = os.getcwd()
    print(data_drop_dir)
    nyt_shsat_df = nyt_web_scrape(data_drop_dir)
    dept_of_ed_df = dept_of_ed_web_scrape(nyt_shsat_df, data_drop_dir, start_idx, debug_flg)

#web_scrape_controller()

final_data_set = pd.read_csv("{}/passnyc-created-data/step2_final_doe_nyt_data.csv".format(input_data_dir))
final_data_set.head()
def features_controller(input_data_dir, output_data_dir):

    output_file_path = "{}/step3_interim_modeling_data.csv".format(output_data_dir)
    
    output_df = pd.read_csv("{}/step2_final_doe_nyt_data.csv".format(input_data_dir))
    
    output_df = clean_percentage_cols(output_df)
    output_df = find_grade_8_flg(output_df)
    output_df = clean_rows_and_cols(output_df)
    output_df = get_addtl_columns(output_df)
    output_df = create_dummy_vars(output_df)
    output_df = get_socrata_data(output_df, input_data_dir)
    output_df = cast_as_bool(output_df)
    output_df = fill_na(output_df)
    output_df.to_csv(output_file_path, index=False)

    print("interim modeling output can be found at: {}".format(output_file_path))
          
    return
import numpy as np
import pandas as pd
import re
import geopy.distance

def cast_as_bool(df):
    
    # sorry...this is messy. just checks to see if there are columns w/data type of float and 1,0
    # then casts as bool    
    for col in df.columns.values:
        if df[col].dtype == "float64":
            if len(df[col].unique()) == 2 \
                and 1 in df[col].unique() \
                and 0 in df[col].unique():

                df[col] = df[col].astype(bool)
            
    return df

def fill_na(df):

    response_vars = ["num_testtakers", "num_offered", "pct_8th_graders_offered", "perc_testtakers", "perc_testtakers_quartile"]
    for response in response_vars:
        if response == "perc_testtakers_quartile":
            na_fill_val = 1
        else:
            na_fill_val = 0

        df[response] = df[response].fillna(value=na_fill_val)    
    
    nobs = float(len(df))
    
    for col in df.columns.values:

        num_nulls = float(df[col].isnull().sum())

        if num_nulls / nobs > .1 or len(df[col].unique()) == 1:

            df = df.drop(col, axis = 1)

        elif num_nulls > 0:
            if df[col].dtype == "object":
                na_fill = df[col].value_counts().idxmax()
            else:
                na_fill = df[col].median()

            df[col] = df[col].fillna(value = na_fill)
    
    #invalid_preds = ["school_name", "dbn", "Address (Full)", "City", "Grades", "Grade Low", "Grade High", "SED Code", "Latitude", "Longitude", "Zip"]
    
    #invalid_preds.extend(response_vars)
#     interim_pred_df = interim_modeling_df.drop(invalid_preds, axis=1)
#     interim_response_df = interim_modeling_df[response_vars]
    return df

def transform_pct(col_string):
    
    if pd.isnull(col_string):
        col_val = col_string
    else:
        result = re.search('(.*)%', col_string)
        col_val = float(result.group(1))
        col_val = col_val / 100

    return col_val

def transform_pct_diff(col_string):

    #test = col_string.extract('^(\+|-)+(.*)%')
    if pd.isnull(col_string):
        col_val = col_string
    else:    
        result = re.search('^(\+|-)+(.*)%', col_string)

        sign = result.group(1) 
        col_val = float(result.group(2))
        positive = True if sign == '+' else False
        col_val = -1 * col_val if positive == False else col_val
        col_val = col_val / 100

    return col_val

def clean_percentage_cols(modeling_df):

    modeling_df_cols = modeling_df.columns.values

    for col in modeling_df_cols:
        df_col = modeling_df[col]

        clean_pct_flg = True if (df_col.dtype == object) and (df_col.str.contains('%').any()) else False
        if clean_pct_flg:

            # reason why escape char \ is used is bc of regex underneath the hood of Series.str.contains
            perc_diff_flg = True if (df_col.str.contains('\+').any()) and (df_col.str.contains('-').any()) else False
            
            if perc_diff_flg == True:
                df_col = df_col.apply(transform_pct_diff)
            else:
                df_col = df_col.apply(transform_pct)
        modeling_df[col] = df_col
    return modeling_df

def find_grade_8_flg(df):

    bool_series = df.apply(lambda row: True if '8' in str(row['Grades']) else False, axis=1)
    df['grade_8_flg'] = bool_series

    return df

def clean_rows_and_cols(df):

    # these schools were established in last year or two, and do not yet have 8th graders
    dbns_to_remove = ["15K839", "03M291", "84X492", "84X460", "28Q358"]
    df = df[~df['dbn'].isin(dbns_to_remove)]
    
    #TODO: use config to pull years and create incoming_state_score_cols in a better way
    incoming_state_score_cols = [
        "incoming_ela_level_1_2017_n",
        "incoming_ela_level_2_2017_n",
        "incoming_ela_level_3_2017_n",
        "incoming_ela_level_4_2017_n",
        "incoming_math_level_1_2017_n",
        "incoming_math_level_2_2017_n",
        "incoming_math_level_3_2017_n",
        "incoming_math_level_4_2017_n"
    ]

    for state_score_col in incoming_state_score_cols:
        df[state_score_col] = df[state_score_col].replace(to_replace="N < 5", value=0)
        df[state_score_col] = df[state_score_col].astype('float')

    nobs = float(len(df))

    # remove schools that don't have 8th graders taking the SHSAT
    df = df[df["grade_8_flg"] == True]

    # remove columns with > 25% nulls
    for col_name in df.columns.values:
        
        col_nulls = float(df[col_name].isnull().sum())
        perc_nulls = col_nulls / nobs
        
        if perc_nulls > 0.25:
            df = df.drop(col_name, axis=1)

    # remove schools that don't have 8th grade enrollment    
    df = df.dropna(axis=0, subset=["grade_8_2017_enrollment"])

    return df

def create_dummy_vars(df):

    categorical_cols = [
        "Community School?",
        "Rigorous Instruction Rating",
        "Collaborative Teachers Rating",
        "Supportive Environment Rating",
        "Effective School Leadership Rating",
        "Strong Family-Community Ties Rating",
        "Trust Rating",
        "Student Achievement Rating",
        "borough"
    ]

    ref_val_dict = {
        "Rigorous Instruction Rating" : "Meeting Target",
        "Collaborative Teachers Rating" : "Meeting Target",
        "Supportive Environment Rating" : "Meeting Target",
        "Effective School Leadership Rating" : "Meeting Target",
        "Strong Family-Community Ties Rating" : "Meeting Target",
        "Trust Rating" : "Meeting Target",
        "Student Achievement Rating" : "Meeting Target"
    }
    for cat_col in categorical_cols:

        dummy_df = pd.get_dummies(df[cat_col], prefix=cat_col, dummy_na=True)
        dummy_df = dummy_df.astype('float')
        df = pd.concat([df, dummy_df], axis=1)

        drop_val = ref_val_dict.get(cat_col, None)
        if drop_val is None:
            drop_val = df.groupby([cat_col]).size().idxmax()
        
        drop_col = "{}_{}".format(cat_col, drop_val)
        
        df = df.drop(drop_col, axis=1)

    return df

def get_socrata_data(df, data_dir):
    print("--getting socrates data to incorporate into modeling")
    disc_funding_data_dir = data_dir.replace('passnyc-created-data', 'new-york-city-council-discretionary-funding')
    school_safety_df = data_dir.replace('passnyc-created-data', 'ny-2010-2016-school-safety-report')
    df = get_disc_funding_by_zip(disc_funding_data_dir, df)
    df = get_school_safety_by_dbn(school_safety_df, df)
    
    return df

def get_school_safety_by_dbn(data_dir, df):
    
    school_safety_filename = '2010-2016-school-safety-report.csv'
    school_safety_df = pd.read_csv("{}/{}".format(data_dir, school_safety_filename))
    consolidated_loc_df = school_safety_df[school_safety_df["DBN"].isnull()]
    
    cols_to_fill = ["Major N", "Oth N", "NoCrim N", "Prop N", "Vio N", 
        "AvgOfMajor N", "AvgOfOth N", "AvgOfNoCrim N", "AvgOfProp N", "AvgOfVio N"] 
    
    consolidated_locations = consolidated_loc_df["Building Name"].unique()
    school_yrs = consolidated_loc_df["School Year"].unique()
    
    consolidated_loc_data = {}
    for location in consolidated_locations:
        consolidated_loc_data[location] = {}
        loc_df = consolidated_loc_df[consolidated_loc_df["Building Name"] == location]

        for year in school_yrs:    
            loc_yr_row = loc_df[loc_df["School Year"] == year]  
            if len(loc_yr_row) == 0:
                # no data for that consolidated loc in that particular school yr
                continue
            elif len(loc_yr_row) > 1:
                raise ValueError("duplicate data found for {} / {}".format(location, year))
            loc_yr_dict = get_loc_yr_data(loc_yr_row, cols_to_fill)
            consolidated_loc_data[location][year] = loc_yr_dict    
    dbn_crimes_df = pd.DataFrame()
    school_safety_df.head()

    for idx, row in school_safety_df.iterrows():

        dbn = row['DBN']
        dbn_nan_flg = isinstance(dbn, float) and np.isnan(dbn)
        building_name = row['Building Name']
        bldg_nan_flg = isinstance(building_name, float) and np.isnan(building_name)
        if dbn_nan_flg == False and bldg_nan_flg == False:
            dbn = dbn.strip()
            school_yr = row['School Year']      
            if building_name not in consolidated_loc_data.keys():
                # no data for consolidated location
                continue
            if school_yr not in consolidated_loc_data[building_name].keys():
                # no data for consolidated location in that school year
                continue
            loc_yr_data = consolidated_loc_data[building_name][school_yr]

            #print([row[col] for col in cols_to_fill])
            for col in cols_to_fill:
                row[col] = loc_yr_data[col]

        dbn_crimes_df = dbn_crimes_df.append(row)
    dbn_crimes_df = dbn_crimes_df[~dbn_crimes_df["DBN"].isnull()]
    dbn_crimes_df = dbn_crimes_df.groupby(['DBN'])[cols_to_fill].agg('sum').reset_index()

    output_df = pd.merge(df, dbn_crimes_df, left_on='dbn', right_on='DBN', how='left').drop("DBN",axis=1)
    
    for col in cols_to_fill:
        col_median = output_df[col].median()
        output_df[col] = output_df[col].fillna(value=col_median)

    numerator_cols = ["Major N", "Oth N", "NoCrim N", "Prop N", "Vio N"]
    for numerator in numerator_cols:
        denominator = "AvgOf{}".format(numerator)
        new_col_name = "{}_proportion".format(numerator)
        output_df[new_col_name] = output_df[numerator].astype(float) / output_df[denominator].astype(float)

    return output_df
    
def get_loc_yr_data(row, cols_to_fill):
    
    row_dict = {}
    for col in cols_to_fill:
        row_dict[col] = row[col].values[0]

    return row_dict

def get_disc_funding_by_zip(data_dir, df):
    
    discretionary_funding_filename = 'new-york-city-council-discretionary-funding-2009-2013.csv'
    disc_fund_df = pd.read_csv("{}/{}".format(data_dir, discretionary_funding_filename))
    
    disc_fund_df = disc_fund_df[disc_fund_df["Status "].isin(["Cleared", "Pending"])]
    disc_fund_df = disc_fund_df[disc_fund_df["Postcode"].notnull()]
    disc_fund_df["zip"] = disc_fund_df.apply(clean_zip, axis=1)

    disc_fund_df = disc_fund_df[disc_fund_df['zip'].apply(lambda x: str(x).isdigit())]
    disc_fund_df["zip"] = disc_fund_df["zip"].astype(int)
    disc_fund_df = disc_fund_df[disc_fund_df['zip'].apply(lambda x: len(str(x)) == 5)]
    
    disc_funds_by_zip = disc_fund_df.groupby(['zip'])['Amount '].agg('sum').to_frame().reset_index()
    disc_funds_by_zip.columns = ["zip", "discretionary_funding"]
    disc_funds_by_zip["discretionary_funding"] = disc_funds_by_zip["discretionary_funding"].astype(float)
    
    output_df = pd.merge(df, disc_funds_by_zip, left_on='Zip', right_on='zip', how='left').drop("zip",axis=1)
    
    return output_df

def clean_zip(row):

    raw_zip = row["Postcode"]
    cleaned_zip = raw_zip.split("-")[0]
    return cleaned_zip

def get_addtl_columns(df):

    df = make_continuous_categorical(df)
    df = get_addtl_response_vars(df)
    df['borough'] = df.apply(get_borough, axis=1)

    dist_df = df.apply(get_dist_from_specialized_schools, axis=1)
    df = pd.concat([df, dist_df], axis=1)

    return df

def get_dist_from_specialized_schools(row):
   
    # this only captures distance from feeder school to one of big three specialized schools.
    # however, other specialized schools are incl in specialized_school_long_lat dictionary, if needed
    row_long_lat = (float(row['Latitude']), float(row['Longitude']))

    specialized_school_long_lat = {
        "bronx_hs_of_science" : (40.87833, -73.89083),
        "brooklyn_latin_school" : (40.705, -73.9388889),
        "brooklyn_tech_hs" : (40.6888889, -73.9766667),
        "hs_for_math_sci_eng" : (40.8215, -73.9490),
        "hs_of_amer_studies" : (40.8749, -73.8952),
        "queens_hs_for_sci": (40.699, -73.797),
        "staten_island_tech" : (40.5676, -74.1181),
        "stuyvesant_hs" : (40.7178801, -74.0137509)
    }
    
    big_three_schools = ["bronx_hs_of_science", "brooklyn_tech_hs", "stuyvesant_hs"]

    row = {}
    for specialized_school, specialized_long_lat in specialized_school_long_lat.items():
        if specialized_school in big_three_schools:
            row["dist_to_{}".format(specialized_school)] = geopy.distance.vincenty(row_long_lat, specialized_long_lat).miles       
    row["min_dist_to_big_three"] = row[min(row, key=row.get)]
    return pd.Series(row)

def get_addtl_response_vars(df):
    
    perc_testtakers = df["num_testtakers"].astype(float) / df["grade_8_2017_enrollment"].astype(float)
    df["perc_testtakers"] = perc_testtakers
    df["perc_testtakers"] = df.apply(lambda row: 1 if row["perc_testtakers"] > 1 else row["perc_testtakers"], axis=1)
    perc_testtakers_quantiles = perc_testtakers.quantile([0.25, 0.5, 0.75])
    quartile_1_max = perc_testtakers_quantiles[0.25]
    quartile_2_max = perc_testtakers_quantiles[0.5]
    quartile_3_max = perc_testtakers_quantiles[0.75]

    df["perc_testtakers_quartile"] = np.nan
    df["perc_testtakers_quartile"] = np.where(df["perc_testtakers"] <= quartile_1_max, 1, df["perc_testtakers_quartile"])
    df["perc_testtakers_quartile"] = np.where(
        (df["perc_testtakers"] > quartile_1_max) & (df["perc_testtakers"] <= quartile_2_max), 
        2, df["perc_testtakers_quartile"])
    df["perc_testtakers_quartile"] = np.where(
        (df["perc_testtakers"] > quartile_2_max) & (df["perc_testtakers"] <= quartile_3_max), 
        3, df["perc_testtakers_quartile"])
    df["perc_testtakers_quartile"] = np.where(df["perc_testtakers"] > quartile_3_max, 4, df["perc_testtakers_quartile"])    
    
    return df

def make_continuous_categorical(df):
    
    binary_cols = {"econ_need_index_2017_city_diff" : 0}

    for col_to_xfrom, cutoff in binary_cols.items():
        new_col_name = "{}_binary".format(col_to_xfrom)
        df[new_col_name] = np.nan
        df[new_col_name] = np.where(df[col_to_xfrom].astype(float) >= cutoff, True, False)
        
    return df

def get_borough(row):
    
    # taken from NYC dept of health: https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm
    borough_zip_dict = {
        "bronx" : [
            10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, \
            10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, \
            10475, 10461, 10462, 10464, 10465, 10472, 10473
        ],
        "brooklyn": [
            11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, \
            11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, \
            11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, \
            11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, \
            11237
        ],
        "manhattan": [
            10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, \
            10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, \
            10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, \
            10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, \
            10024, 10025, 10031, 10032, 10033, 10034, 10040, 10282
        ],
        "queens": [
            11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, \
            11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, \
            11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, \
            11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, \
            11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, \
            11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, \
            11368, 11369, 11370, 11372, 11373, 11377, 11378
        ],
        "staten_island" : [
            10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312, 10301, \
            10304, 10305, 10314, 10311
        ]
    } 

    school_zip = row["Zip"]

    school_boro = None
    for boro, borough_zip_list in borough_zip_dict.items():
        if school_zip in borough_zip_list:
            school_boro = boro
            break
    if school_boro is None:
        school_boro = "other"

    return school_boro
cwd = os.getcwd()
input_data_dir = "{}/passnyc-created-data".format(cwd.replace('/working', '/input'))
print(input_data_dir)
features_controller(input_data_dir, cwd)
# the line below should work if you have ran the above snippet. For consistency, I'll bring in the data frame from the ../input directory already uploaded
#interim_modeling_df = pd.read_csv('/kaggle/working/step3_interim_modeling_data.csv')
interim_modeling_df = pd.read_csv('{}/step3_interim_modeling_data.csv'.format(input_data_dir))
interim_modeling_df.describe()

from sklearn import metrics

def adj_r2_score(lm, y, y_pred):
    adj_r2 = 1 - float(len(y)-1)/(len(y)-len(lm.coef_)-1)*(1 - metrics.r2_score(y,y_pred))
    return adj_r2

def get_pred_and_response_dfs(df):

    invalid_preds = ["school_name", "dbn", "Address (Full)", "City", "Grades", "Grade Low", "Grade High", \
    "SED Code", "Latitude", "Longitude", "Zip"]
    response_vars = ["num_testtakers", "num_offered", "pct_8th_graders_offered", "perc_testtakers", "perc_testtakers_quartile"]
    response_var = "perc_testtakers"
    invalid_preds.extend(response_vars)
    
    # incl dbn and school_name for convenience and to id each row
    # incl enrollment to calculate addtl est'd num students
    response_df_cols = [ "school_name", "dbn", "Longitude", "Latitude", response_var, "num_testtakers","grade_8_2017_enrollment", "pct_poverty_2017_val", "Percent Black / Hispanic", ]
    
    response_df = df[response_df_cols]
    pred_df = df.drop(response_df_cols, axis=1)
    
    return pred_df, response_df, invalid_preds, response_var
import os
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics

def create_single_regressors(input_data_dir, output_data_dir, output_model_summaries_dir):

    interim_modeling_df = pd.read_csv("{}/step3_interim_modeling_data.csv".format(input_data_dir))

    pred_df, response_df, invalid_preds, response_var = get_pred_and_response_dfs(interim_modeling_df)
   
    pred_train, pred_test, response_train, response_test = train_test_split(pred_df, response_df, test_size=0.25, random_state=223)

    single_regression_df = pd.DataFrame()

    for col in pred_train.columns.values:
        if col in invalid_preds:
            continue
        else: 
            categorical = True if pred_train[col].dtype == "object" else False
            pred_rows = get_single_regressor_model(col, pred_train, response_train, pred_test, response_test, response_var, categorical)
            single_regression_df = single_regression_df.append(pred_rows, ignore_index=True)
            
    single_regression_df_cols = ["model", "categorical", "model_r2", "model_adj_r2", "median_absolute_error", "pred_col", "pred_coef"]
    single_regression_df = single_regression_df[single_regression_df_cols]
    output_file_path = "{}/single_regressor_summary.csv".format(output_model_summaries_dir)
    print("dropping single regression CSV to {}".format(output_file_path))
    
    single_regression_df.to_csv(output_file_path, index=False)
    
def get_single_regressor_model(col_name, pred_train, response_train, pred_test, response_test, response_var, categorical_bool):
    model_rows = []
    
    col_list = pred_train.columns.values
    
    if categorical_bool == True:
        col_vars = [var for var in col_list if "{}_".format(col_name) in var]
    else:
        col_vars = [col_name]
    x_train_df = pred_train[col_vars]
    x_test_df = pred_test[col_vars]
    x_model = linear_model.LinearRegression()
    x_results = x_model.fit(x_train_df, response_train[response_var])

    y_predicted_test = x_results.predict(x_test_df)
    med_absolute_error = metrics.median_absolute_error(response_test[response_var], y_predicted_test)
    r2 = metrics.r2_score(response_test[response_var],y_predicted_test)
    adj_r2 = adj_r2_score(x_results, response_test[response_var],y_predicted_test)

    coef_dict = {}
    model_rows = []
    for idx, col in enumerate(x_train_df.columns.values):
        coef_dict[col] = x_results.coef_[idx]
        model_row = {}
        model_row['model'] = col_name
        pred_coef = x_results.coef_[idx]
        model_row['model_r2'] = r2
        model_row['model_adj_r2'] = adj_r2
        model_row['median_absolute_error'] = med_absolute_error
        model_row['pred_col'] = col if categorical_bool == True else None
        model_row['pred_coef'] = pred_coef
        model_row['categorical'] = categorical_bool
        model_rows.append(model_row)

    return model_rows
# we can use the previously-defined input_ and output_data_dir variables to pass into our function create_single_regressors.
# we pass in a third directory for the output of these functions. For the purpose of the kernel, we'll set it to our current working dir
cwd = os.getcwd()
input_data_dir = "{}/passnyc-created-data".format(cwd.replace('/working', '/input'))
output_data_dir = cwd
output_model_summaries_dir = cwd
print(input_data_dir)
print(output_data_dir)
print(output_model_summaries_dir)
create_single_regressors(input_data_dir, cwd, cwd)
output_model_summaries_dir = os.getcwd()
single_regression_df = pd.read_csv("{}/single_regressor_summary.csv".format(output_model_summaries_dir))
single_regression_df.sort_values(by="model_r2", ascending=False)
single_regression_df[single_regression_df["model"].isin(['borough', 'min_dist_to_big_three'])]
socrata_data_cols = ['discretionary_funding', 'Major N', 'Oth N', 'NoCrim N', 'Prop N', 'Vio N', 
                     'Major N_proportion', 'Oth N_proportion', 'NoCrim N_proportion', 'Prop N_proportion', 'Vio N_proportion']
single_regression_df[single_regression_df["model"].isin(socrata_data_cols)].sort_values(by="model_r2", ascending=False)
import os
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics, preprocessing

def create_final_model(input_data_dir, output_data_dir, output_model_summaries_dir):

    interim_modeling_df = pd.read_csv("{}/step3_interim_modeling_data.csv".format(input_data_dir))

    pred_df, response_df, invalid_preds, response_var = get_pred_and_response_dfs(interim_modeling_df)
    
    model_1_pred_train, model_1_pred_test, model_1_response_train, model_1_response_test = \
    train_test_split(pred_df, response_df, test_size=0.5, random_state=223)
    
    model_2_pred_train = model_1_pred_test
    model_2_pred_test = model_1_pred_train
    model_2_response_train = model_1_response_test
    model_2_response_test = model_1_response_train
    
    final_preds = [
    #"school_name", "dbn", # only incl these two for convenience in ID'ing rows
    "Average ELA Proficiency", 
    "pct_math_level_3_or_4_2017_city_diff", 
    "sa_attendance_90plus_2017", 
    "pct_8th_graders_w_hs_credit_2017_city_diff",
    "min_dist_to_big_three"
#     "Collaborative Teachers Rating_Approaching Target",
#     "Collaborative Teachers Rating_Exceeding Target",
#     "Collaborative Teachers Rating_Not Meeting Target",
#     "Collaborative Teachers Rating_nan",
#     "Major N_proportion"
]
    model_1_train_pred_df = model_1_pred_train[final_preds]
    model_1_test_pred_df = model_1_pred_test[final_preds]
    
    model_2_train_pred_df = model_2_pred_train[final_preds]
    model_2_test_pred_df = model_2_pred_test[final_preds]
    
    stdized_model_1_train, stdized_model_1_test = standardize_cols(model_1_train_pred_df, model_1_test_pred_df)
    stdized_model_2_train, stdized_model_2_test = standardize_cols(model_2_train_pred_df, model_2_test_pred_df)
    
    model_1 = linear_model.LinearRegression().fit(stdized_model_1_train, model_1_response_train[response_var])
    model_1_train_predicted = model_1.predict(stdized_model_1_train) 
    model_1_test_predicted = model_1.predict(stdized_model_1_test)
    model_1_response_train["predicted_perc_testtakers"] = model_1_train_predicted
    model_1_response_test["predicted_perc_testtakers"] = model_1_test_predicted
    model_1_coefficients = pd.concat([pd.DataFrame(stdized_model_1_train.columns),pd.DataFrame(np.transpose(model_1.coef_))], axis = 1)
    model_1_coefficients.columns = ["model_1_pred_name", "model_1_coef"]
    model_1_full_train_df = pd.concat([model_1_response_train, stdized_model_1_train], axis = 1)
    model_1_full_test_df = pd.concat([model_1_response_test, stdized_model_1_test], axis = 1)

    model_2 = linear_model.LinearRegression().fit(stdized_model_2_train, model_2_response_train[response_var])
    model_2_train_predicted = model_1.predict(stdized_model_2_train) 
    model_2_test_predicted = model_2.predict(stdized_model_2_test)
    model_2_response_train["predicted_perc_testtakers"] = model_2_train_predicted
    model_2_response_test["predicted_perc_testtakers"] = model_2_test_predicted
    model_2_coefficients = pd.concat([pd.DataFrame(stdized_model_2_train.columns),pd.DataFrame(np.transpose(model_2.coef_))], axis = 1)
    model_2_coefficients.columns = ["model_2_pred_name", "model_2_coef"]
    model_2_full_train_df = pd.concat([model_2_response_train, stdized_model_2_train], axis = 1)
    model_2_full_test_df = pd.concat([model_2_response_test, stdized_model_2_test], axis = 1)
    
    final_train_set = pd.concat([model_1_full_train_df, model_2_full_train_df])
    final_test_set =  pd.concat([model_1_full_test_df, model_2_full_test_df])

    final_train_set.to_csv("{}/full_train_dataset.csv".format(output_data_dir), index = False)
    final_test_set.to_csv("{}/full_test_dataset.csv".format(output_data_dir), index = False)
    model_1_coefficients.to_csv("{}/model_1_coefficients.csv".format(output_model_summaries_dir), index = False)
    model_2_coefficients.to_csv("{}/model_2_coefficients.csv".format(output_model_summaries_dir), index = False)
    
    final_r2 = metrics.r2_score(final_test_set["perc_testtakers"], final_test_set["predicted_perc_testtakers"])
    final_median_abs_err = metrics.median_absolute_error(final_test_set["perc_testtakers"], final_test_set["predicted_perc_testtakers"])
    
    print("\n\nmodel_1_coefficients: ")
    print(model_1_coefficients)
    print("\nmodel_2_coefficients: ")
    print(model_2_coefficients)
    print("\n\n** r2 of entire dataset: {}".format(final_r2))
    print("** median_absolute_error of entire dataset: {}".format(final_median_abs_err))
    
    print("\n\ndropped output to folder: {}".format(output_model_summaries_dir))
    
    
def standardize_cols(train_df, test_df):
    
    standardized_train_df = train_df
    standardized_test_df = test_df
    for pred_col in train_df.columns.values:
        if train_df[pred_col].dtype == "float64":
            scaler = preprocessing.StandardScaler().fit(standardized_train_df[[pred_col]])
            standardized_train_df["{}_stdized".format(pred_col)] = scaler.transform(standardized_train_df[[pred_col]])
            standardized_test_df["{}_stdized".format(pred_col)] = scaler.transform(standardized_test_df[[pred_col]])
            standardized_train_df = standardized_train_df.drop(pred_col, axis=1)
            standardized_test_df = standardized_test_df.drop(pred_col, axis=1)
    return standardized_train_df, standardized_test_df
cwd = os.getcwd()
input_data_dir = "{}/passnyc-created-data".format(cwd.replace('/working', '/input'))
output_data_dir = cwd
output_model_summaries_dir = cwd
create_final_model(input_data_dir, output_data_dir, output_model_summaries_dir)

model_performance_df = pd.read_csv("{}/passnyc-created-data/step4_full_test_dataset.csv".format(cwd.replace('/working', '/input')))
pred_actual_scatter = model_performance_df.plot(kind="scatter", 
    x="perc_testtakers", 
    y="predicted_perc_testtakers",
    c=model_performance_df["pct_poverty_2017_val"].astype(float),
    cmap=plt.get_cmap("coolwarm"), 
    s=100,
    colorbar=True,
    alpha=0.75 ,
    title='Predicted versus Actual Percentage of SHSAT Testtakers (Color denotes % Below Poverty Level)',
    figsize=(20,10))

pred_actual_scatter.set_ylabel("Predicted Percentage of Testtakers")
plt.show()
input_data_dir = os.getcwd().replace('/working', '/input')
model_performance_df = pd.read_csv("{}/passnyc-created-data/step4_full_test_dataset.csv".format(input_data_dir))
nyc_img=mpimg.imread('{}/passnyc-created-data/cropped_neighbourhoods_new_york_city_map.png'.format(input_data_dir))

model_performance_df["predicted_testtakers_all"] = model_performance_df["predicted_perc_testtakers"].astype(float) * model_performance_df["grade_8_2017_enrollment"].astype(float)
model_performance_df["predicted_testtakers_black_hispanic"] = model_performance_df["predicted_testtakers_all"].astype(float) * model_performance_df["Percent Black / Hispanic"].astype(float)
model_performance_df["predicted_testtakers_below_poverty_lvl"] = model_performance_df["predicted_testtakers_all"].astype(float) * model_performance_df["pct_poverty_2017_val"].astype(float)
model_performance_df["actual_testtakers_black_hispanic"] = model_performance_df["num_testtakers"].astype(float) * model_performance_df["Percent Black / Hispanic"].astype(float)
model_performance_df["actual_testtakers_below_poverty_lvl"] = model_performance_df["num_testtakers"].astype(float) * model_performance_df["pct_poverty_2017_val"].astype(float)
model_performance_df["diff_num_testtakers_all"] =  model_performance_df["predicted_testtakers_all"] - model_performance_df["num_testtakers"]
model_performance_df["diff_num_testtakers_black_hispanic"] = model_performance_df["predicted_testtakers_black_hispanic"] - model_performance_df["actual_testtakers_black_hispanic"] 
model_performance_df["diff_num_testtakers_below_poverty_lvl"] = model_performance_df["predicted_testtakers_below_poverty_lvl"] - model_performance_df["actual_testtakers_below_poverty_lvl"] 
model_performance_df["percentile_diff_num_testtakers_all"] = model_performance_df.apply(lambda row: float(scipy.stats.percentileofscore(model_performance_df["diff_num_testtakers_all"], row["diff_num_testtakers_all"])) / float(100), axis=1)
model_performance_df["percentile_diff_num_black_hispanic"] = model_performance_df.apply(lambda row: float(scipy.stats.percentileofscore(model_performance_df["diff_num_testtakers_black_hispanic"], row["diff_num_testtakers_black_hispanic"])) / float(100), axis=1)
model_performance_df["percentile_diff_num_below_poverty_lvl"] = model_performance_df.apply(lambda row: float(scipy.stats.percentileofscore(model_performance_df["diff_num_testtakers_black_hispanic"], row["diff_num_testtakers_below_poverty_lvl"])) / float(100), axis=1)

model_performance_df["positive_diff_num_testtakers"] = model_performance_df.apply(lambda row: True if row["diff_num_testtakers_all"] > 0 else False, axis=1)
model_performance_df["addtl_testtakers_label"] = model_performance_df.apply(lambda row: "{}: {} {} testtakers predicted".format(row["school_name"], int(row["diff_num_testtakers_all"]), "fewer" if int(row["diff_num_testtakers_all"]) < 0 else "additional") , axis=1)
model_performance_df["addtl_testtakers_blk_hisp_label"] = model_performance_df.apply(lambda row: "{}: {} {} testtakers predicted".format(row["school_name"], int(row["diff_num_testtakers_black_hispanic"]), "fewer" if int(row["diff_num_testtakers_black_hispanic"]) < 0 else "additional") , axis=1)
model_performance_df["addtl_testtakers_below_poverty_lvl_label"] = model_performance_df.apply(lambda row: "{}: {} {} testtakers predicted".format(row["school_name"], int(row["diff_num_testtakers_below_poverty_lvl"]), "fewer" if int(row["diff_num_testtakers_below_poverty_lvl"]) < 0 else "additional") , axis=1)

new_int_cols = ["predicted_testtakers_all", "predicted_testtakers_black_hispanic", \
    "predicted_testtakers_below_poverty_lvl", "actual_testtakers_black_hispanic", \
    "actual_testtakers_below_poverty_lvl", "diff_num_testtakers_all", \
    "diff_num_testtakers_black_hispanic", "diff_num_testtakers_below_poverty_lvl"
] 
model_performance_df[new_int_cols] = model_performance_df[new_int_cols].astype(int)
model_performance_cols_to_show = [
    "school_name", "dbn", "perc_testtakers", "predicted_perc_testtakers", "diff_num_testtakers_all",
    "predicted_testtakers_all", "num_testtakers",
    "predicted_testtakers_black_hispanic", "actual_testtakers_black_hispanic", 
    "predicted_testtakers_below_poverty_lvl", "actual_testtakers_below_poverty_lvl",
]
model_performance_df.sort_values(by="diff_num_testtakers_all", ascending = False )

positive_df = model_performance_df[model_performance_df["positive_diff_num_testtakers"] == True]
positive_df.plot(kind="scatter", 
    x="Longitude", 
    y="Latitude", 
    c=positive_df["pct_poverty_2017_val"].astype(float), 
    s=positive_df["diff_num_testtakers_all"].astype(float) * 25, 
    cmap=plt.get_cmap("coolwarm"), 
    title='Additional Expected Number of Testtakers (Color denotes % Below Poverty Level)',
    figsize=(16.162,16),
    alpha=0.5)
plt.imshow(nyc_img, 
           extent=[model_performance_df["Longitude"].min() - .01 , model_performance_df["Longitude"].max() + .01, model_performance_df["Latitude"].min() - .01, model_performance_df["Latitude"].max() + .01], 
           alpha = 0.25)
plt.show()
pos_data = [
    {
        'x': positive_df["Longitude"],
        'y': positive_df["Latitude"],
        'text': positive_df["addtl_testtakers_label"],
        'mode': 'markers',
        'marker': {
            'color': positive_df["pct_poverty_2017_val"].astype(float),
            'size': positive_df["diff_num_testtakers_all"].astype(float) * 1.2,
            'showscale': True,
            'colorscale':'RdBu',
            'opacity':0.5
        }
    }
]

layout= go.Layout(
    autosize=False,
    width=900,
    height=750,
    title= 'Additional Expected Number of Testtakers  (Color denotes % Below Poverty Level)',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    ))
fig=go.Figure(data=pos_data,layout=layout)
plotly.offline.iplot(fig, filename='scatter_hover_labels')
positive_df.plot(kind="scatter", 
    x="Longitude", 
    y="Latitude", 
    c=positive_df["pct_poverty_2017_val"].astype(float), 
    s=positive_df["diff_num_testtakers_black_hispanic"].astype(float) * 25, 
    cmap=plt.get_cmap("coolwarm"), 
    title='Additional Expected Number of Black/Hispanic Testtakers (Color denotes % Below Poverty Level)',
    figsize=(16.162,16),
    alpha=0.5)
plt.imshow(nyc_img, 
           extent=[model_performance_df["Longitude"].min() - .01 , model_performance_df["Longitude"].max() + .01, model_performance_df["Latitude"].min() - .01, model_performance_df["Latitude"].max() + .01], 
           alpha = 0.25)
plt.show()
pos_data = [
    {
        'x': positive_df["Longitude"],
        'y': positive_df["Latitude"],
        'text': positive_df["addtl_testtakers_blk_hisp_label"],
        'mode': 'markers',
        'marker': {
            'color': positive_df["pct_poverty_2017_val"].astype(float),
            'size': positive_df["diff_num_testtakers_black_hispanic"].astype(float) * 1.2,
            'showscale': True,
            'colorscale':'RdBu',
            'opacity':0.5
        }
    }
]

layout= go.Layout(
    autosize=False,
    width=850,
    height=750,
    title= 'Additional Expected Number of Black/Hispanic Testtakers (Color denotes % Below Poverty Level)',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    ))
fig=go.Figure(data=pos_data,layout=layout)
plotly.offline.iplot(fig, filename='scatter_hover_labels')
positive_df.plot(kind="scatter", 
    x="Longitude", 
    y="Latitude", 
    c=positive_df["pct_poverty_2017_val"].astype(float), 
    s=positive_df["diff_num_testtakers_below_poverty_lvl"].astype(float) * 25, 
    cmap=plt.get_cmap("coolwarm"), 
    title='Additional Expected Number of Testtakers Below Poverty Level (Color denotes % Below Poverty Level)',
    figsize=(16.162,16),
    alpha=0.5)
plt.imshow(nyc_img, 
           extent=[model_performance_df["Longitude"].min() - .01 , model_performance_df["Longitude"].max() + .01, model_performance_df["Latitude"].min() - .01, model_performance_df["Latitude"].max() + .01], 
           alpha = 0.25)
plt.show()
pos_data = [
    {
        'x': positive_df["Longitude"],
        'y': positive_df["Latitude"],
        'text': positive_df["addtl_testtakers_below_poverty_lvl_label"],
        'mode': 'markers',
        'marker': {
            'color': positive_df["pct_poverty_2017_val"].astype(float),
            'size': positive_df["diff_num_testtakers_below_poverty_lvl"].astype(float) * 1.2,
            'showscale': True,
            'colorscale':'RdBu',
            'opacity':0.5
        }
    }
]

layout= go.Layout(
    autosize=False,
    width=850,
    height=750,
    title= 'Additional Expected Number of Testtakers below Poverty Level (Color denotes % Below Poverty Level)',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    ))
fig=go.Figure(data=pos_data,layout=layout)
plotly.offline.iplot(fig, filename='scatter_hover_labels')
prediction_error = abs(model_performance_df["predicted_perc_testtakers"] - model_performance_df["perc_testtakers"])
pred_error_quantiles = prediction_error.quantile([.1,.25,.5, .75,.95])
abs_pred_err_95_perc = pred_error_quantiles[.95]
model_performance_df[["school_name","grade_8_2017_enrollment", "perc_testtakers", "predicted_perc_testtakers"]][abs(model_performance_df["predicted_perc_testtakers"] - model_performance_df["perc_testtakers"]) > abs_pred_err_95_perc].sort_values(by="perc_testtakers")