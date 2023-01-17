%matplotlib inline
import pandas as pd
import numpy as np
import geopandas as gpd 
from os import listdir
import os.path
import urllib
import zipfile
import os
import requests
from ftplib import FTP


import matplotlib.pyplot as plt
import matplotlib
import pysal as ps
import plotly.offline as offline
import plotly.graph_objs as go
import cufflinks as cf
from shapely.geometry import Point
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools
offline.init_notebook_mode(connected=True)

from numpy import arange,array,ones
from scipy import stats


def fix_police_districts(gdf,incident, set_limit):
    temp=test_police_district_as_is(gdf, incident)
    if temp[0]>set_limit:
        temp=test_police_district_takin_first_item(gdf, incident)
        if temp[0]>set_limit:
            temp=test_lookup_strings_containing_substring(gdf, incident) 
    return temp
    
    
def rename_district_columns(df_dictionary, police_gdf_col_name=None):
    set_limit=2
    incident=df_dictionary['incident_df'].copy()
    gdf=df_dictionary['police_gdf'].copy()
    if 'LOCATION_DISTRICT' in list(incident):
        incident['DISTRICT']=incident['LOCATION_DISTRICT']
    else:
        incident['DISTRICT']=0
    if (police_gdf_col_name!=None) & (police_gdf_col_name in list(gdf)):
        
        gdf['DISTRICT']=gdf[police_gdf_col_name]
        
    if 'DISTRICT' not in list(gdf):
        for item in list(gdf):
            if item!='geometry':
                temp_df=gdf.copy()
                temp_df.rename(columns={item:'DISTRICT'},inplace=True)
                try:
                    temp_df['DISTRICT']=temp_df['DISTRICT'].astype(incident['DISTRICT'].dtype)
                    temp_df['DISTRICT']=temp_df['DISTRICT'].astype(str)
                    incident['DISTRICT']=incident['DISTRICT'].astype(str)
                    set_limit=1
                except:
                    set_limit=2
                working=fix_police_districts(temp_df,incident, set_limit)
                if working[0]<=set_limit:
                    df_dictionary['police_gdf']=working[1]
                    df_dictionary['incident_df']=working[2]
                    return df_dictionary
    else:
        temp_df=gdf.copy()
        working=fix_police_districts(temp_df,incident, set_limit)
        df_dictionary['police_gdf']=working[1]
        df_dictionary['incident_df']=working[2]
        return df_dictionary
    
    return df_dictionary

def test_police_district_as_is(gdf, incident): 
    gdf_set=set(gdf.DISTRICT)
    incident_set=set(incident.DISTRICT)
    return [len(gdf_set - incident_set), gdf, incident]

def test_police_district_takin_first_item(gdf, incident):
    
    temp_df=incident.copy()
    temp_df['DISTRICT']=temp_df['DISTRICT'].astype(str)
    temp_df['DISTRICT']=temp_df['DISTRICT'].str.split(' ').str[0]
    gdf_set=set(gdf.DISTRICT)
    incident_set=set(temp_df.DISTRICT)
    
    return [len(gdf_set - incident_set), gdf, temp_df]
        
def test_lookup_strings_containing_substring(gdf, incident):
    temp_df=incident.copy()
    incident_list=list(set(temp_df['DISTRICT']))
    
    gdf_list=list(set(gdf['DISTRICT']))
    df=pd.DataFrame(columns={'DISTRICT','GDF'})
    for item in incident_list:
        gdf_district=np.nan
        incident_district=item
        temp_list=[x for x in gdf_list if str(item) in x]
        if len(temp_list)>0:
            gdf_district=temp_list[0]
        else: gdf_district=item
        temp_df2=pd.DataFrame(data=[[gdf_district,incident_district]],columns={'DISTRICT','GDF'})
        df=df.append(temp_df2)
    temp_df['DISTRICT']=temp_df['DISTRICT'].astype(str)
    df['DISTRICT']=df['DISTRICT'].astype(str)
    temp_df=pd.merge(temp_df,df,how='left',left_on='DISTRICT',right_on='DISTRICT')
    temp_df.drop(columns=['DISTRICT'],inplace=True)
    temp_df.rename(columns={'GDF':'DISTRICT'},inplace=True)
    temp=test_police_district_as_is(gdf, temp_df)
    return temp

#plotly functions

def draw_scatter(final_df,y,x,size, fig_plot, row_num,col_num, color, size_division=500):
    size_division=final_df[size].max()/40
    normalized_x=final_df[x]
    normalized_y=final_df[y]
    #normalized_x=(final_df[x]-final_df[x].min())/(final_df[x].max()-final_df[x].min())*100
    #normalized_y=(final_df[y]-final_df[y].min())/(final_df[y].max()-final_df[y].min())*100
    slope, intercept, r_value, p_value, std_err = stats.linregress(normalized_x,normalized_y)
    line = slope*normalized_x+intercept
    trace0=go.Scatter(
            x=normalized_x,
            y=normalized_y,
            text=final_df['Police_Area_Name'],
            mode='markers',
            marker=dict(color=color,size=final_df[size]/size_division))
    fig_plot.append_trace(trace0,row_num,col_num)
    return fig_plot

def draw_map(df, col_name, fig_plot,row_num,col_num):
    data=[]
    #text='Police Zone: {} <br>Total Population: {} <br>{} Population: {} ({}%)'.format(row[police_zone_col_name],row['Total_Pop'],col_name, row[col_name+'_Pop'],row[final_col_name]),
    text='test2'
    norm=matplotlib.colors.Normalize(0,max(df[col_name]))

    for index, row in df.iterrows():

        X, Y= row['geometry'].exterior.coords.xy
        X=list(X)
        Y=list(Y)
        Z='grey'
        trace0=go.Scatter(
            x=X,
            y=Y,
            mode='lines',
            fill='toself',
            fillcolor=matplotlib.colors.rgb2hex(matplotlib.cm.Reds(norm(row[col_name]))),
            text='Police Zone: {} <br>{}: {}'.format(row['Police_Area_Name'],col_name, row[col_name]),
            
            name=row['Police_Area_Name'],
            hoverinfo='text',
            line=dict(width=0.5,color=Z))
        data.append(trace0)
        fig_plot.append_trace(trace0,row_num,col_num)

    
    
    #fig = go.Figure(data=data, layout=layout)
    #fig_plot.append_trace(fig,row_num,col_num)
    return fig_plot

def massage_acs(acs):
    acs=drop_acs_columns(acs)
    acs_stack=stack_acs(acs)
    acs_atack=replace_acs(acs_stack)
    acs_stack=rename_acs_stack(acs_stack)
    acs_stack.Id=acs_stack.Id.str.replace('Estimate; ','')
    acs_stack.Id=acs_stack.Id.str.replace('Total; ','')
    acs_stack['TEMP']=acs_stack.Id.str.split('; ')
    acs_stack['TEMP2']=acs_stack['TEMP'].apply(lambda x: ' - '.join(x[-1].split(' - ')[:-1]) if ' - ' in x[-1] else x[-1])
    conditions=[acs_stack['TEMP2'].str.startswith('AGE'),
                acs_stack['TEMP2'].str.contains('POVERTY STATUS IN THE PAST 12 MONTHS'),
                ]
    option_list=['Population 16 years and over',
                 'Population 16 years and over',
                ]
    acs_stack['TEMP2']=np.select(conditions,option_list,acs_stack['TEMP2'])
    acs_stack=pd.merge(acs_stack,acs_stack[['CENSUS_TRACT', 'Id', 'COUNT']], how='left', left_on=['CENSUS_TRACT', 'TEMP2'], right_on=['CENSUS_TRACT','Id'])
    acs_stack=acs_stack[['CENSUS_TRACT','Id_x','COUNT_x','COUNT_y']]
    acs_stack.rename(columns={'Id_x':'CATEGORY', 'COUNT_x':'COUNT','COUNT_y':'TOTAL'}, inplace=True)
    acs_stack['COUNT']=pd.to_numeric(acs_stack['COUNT'], errors='coerce')
    acs_stack['TOTAL']=pd.to_numeric(acs_stack['TOTAL'], errors='coerce')
    acs_stack['TOTAL'].fillna(acs_stack['COUNT'],inplace=True)
    return acs_stack

def drop_acs_columns(acs):
    acs.set_index('Id2',inplace=True)
    acs.drop(columns=['Id','Geography'], inplace=True)
    return acs

def stack_acs(acs):
    acs_stack=acs.stack().reset_index()
    acs_stack.rename(columns={'Id2':'CENSUS_TRACT', 'level_1':'Id',0:'COUNT'},inplace=True)
    acs_stack=acs_stack[(acs_stack['Id'].str.contains('Estimate'))]
    return acs_stack

def replace_acs(acs):
    acs.Id=acs.Id.str.replace(' - Total population','')
    acs.Id=acs.Id.str.replace(' - SEX','')
    return acs

def rename_acs_stack(acs):
    acs.rename(columns={'GEO.id2':'CENSUS_TRACT', 
                              'level_1':'ACS_CODE',
                              0:'COUNT'},inplace=True)
    return acs

def get_length_of_split_cell(acs,col_name):
    acs['len']=acs[col_name].apply(lambda x: len(x))
    ###
    if max(acs['len'])>1:
        col_list=['ACS_CODE_1']
    else:
        col_list=None
    return col_list

#check for the required files
def ask_for_dept(path,file):
    #department_directory_list=[]
    #counter=1
    #for file in listdir(path):
    #    space=''
    #    if file.startswith('Dept_'):
    #        if counter<10:
    #            space='  '
    #        elif counter<100:
    #            space=' '
    #        print('{}.{} {}'.format(counter,space,file))
    #        department_directory_list.append(path+file+'/')
    #        counter+=1
    #dept=int(input('Select the number of the Department you want to review: '))-1
    dept=path+file+'/'
    return dept

def check_departments_sub_folders(dept_path):
    df_dictionary={}
    for file in listdir(dept_path):
        if  not file.startswith('.'):
            df_dictionary=check_file(dept_path,file,df_dictionary)
    return df_dictionary

def check_file(path,file,df_dictionary):
    
    if  not file.startswith('.'):
        
        if file=='acs.csv':
            title='acs_merged_df'
        elif (file.endswith('.csv')) & ('ACS_' not in path):
            title='incident_df'
        elif file.endswith('ann.csv'):
            title='acs_data'
        elif file.endswith('.shp'):
            title='police_gdf'
        elif (('ACS' in file) & ('.' not in file)) | (file.endswith('Shapefiles')):
            path=path+file+'/'
            title='additional_files'
            for sub_file in listdir(path):
                df_dictionary=check_file(path,sub_file,df_dictionary)
        else: title='additional_files'
        if title not in df_dictionary:
            df_dictionary[title]=[]
        df_dictionary[title].append(path+file)
       
    return df_dictionary

def check_for_census_shapefile(path,state_cd,city_cd,df_dictionary, census_path):
    data=[]
    download_shapefile='n'
    #census_path=path+'census_tracts/'
    #if not os.path.isdir(census_path):
    #    os.mkdir(census_path)
    if not os.path.isdir(census_path+str(state_cd)+'/'):
        download_shapefile=input('Census Tract shapefile does not exist. Would you like to download it now? ')
    #if download_shapefile.lower()=='y':
    #    download_census_tract_shapefile(path,state_cd,city_cd)
    for item in listdir(census_path+str(state_cd)+'/'):
        
        if item.endswith('.shp'):
            data.append(census_path+str(state_cd)+'/'+item)
    df_dictionary['census_gdf']=data

    return df_dictionary
    
def get_state_cd(dept_path):
    state_cd=dept_path.split('_')[1].split('-')[0]
    city_cd=int(dept_path.split('_')[1].split('-')[1].split('/')[0])
    return [state_cd, city_cd]
    
def download_census_tract_shapefile(path,state_cd,city_cd):
    census_path=path+'census_tracts/'
    dept_path=path+'Dept_'+str(state_cd).zfill(2)+'-'+str(city_cd).zfill(5)+'/'+str(state_cd).zfill(2)+'-'+str(city_cd).zfill(5)+'_ACS_data/'
    for file in listdir(dept_path+listdir(dept_path)[1]+'/'):
        if file.lower().endswith('ann.csv'):
            temp_df=pd.read_csv(dept_path+listdir(dept_path)[1]+'/'+file,nrows=2)
            census_cd=(int(temp_df.iloc[1][1][:2]))
            download_file_name='tl_2018_'+str(int(census_cd)).zfill(2)+'_tract'
            url = 'ftp://ftp2.census.gov/geo/tiger/TIGER2018/TRACT/'+download_file_name+'.zip'
            #print(url)
            response=urllib.request.urlopen(url)
            print('got response')
            zipped_data=response.read()
            print('got zipped data')
            output=open(census_path+str(state_cd)+'.zip','wb')
            output.write(zipped_data)
            print('saved zip file at {}'.format(census_path+str(state_cd)+'.zip'))
            zip_ref = zipfile.ZipFile(census_path+str(state_cd)+'.zip', 'r')
            zip_ref.extractall(census_path+str(state_cd))
            zip_ref.close()
            os.remove(census_path+str(state_cd)+'.zip')
    return None

def clean_file_dictionary(path,df_dictionary):
    if 'acs_merged_df' in df_dictionary.keys():
        del df_dictionary['acs_data']
    if os.path.isfile(path+'crs_references.csv'):
        df_dictionary['crs_reference_df']=[path+'crs_references.csv']
    del df_dictionary['additional_files']
    return df_dictionary

def load_dataframes(df_dictionary):
    #df=pd.read_csv(df_dictionary['acs_data'][0])
    for title in df_dictionary.keys():
        
        if title=='acs_data':
            temp_dict={}
            for file in range(len(df_dictionary[title])):
                
                
                new_title=df_dictionary[title][file].split('_ACS_')[2].split('/')[0]
                
                temp_dict[new_title]=pd.read_csv(df_dictionary[title][file],skiprows=1)
        
        else:
            selected_file_num=check_number_of_files(df_dictionary,title)
            if title.lower().endswith('_gdf'):         
                df_dictionary[title]=gpd.read_file(df_dictionary[title][selected_file_num]) 
            elif title.lower()=='crs_reference_df':
                df_dictionary[title]=pd.read_csv(df_dictionary[title][selected_file_num],skiprows=0)
            else: df_dictionary[title]=pd.read_csv(df_dictionary[title][selected_file_num],skiprows=[1])
    
    if 'acs_data' in df_dictionary.keys():
        df_dictionary['acs_data_to_be_tidied']=temp_dict
        del df_dictionary['acs_data']
            
    return df_dictionary
        
def check_number_of_files(df_dictionary,title):
    selected_file=0
    if len(df_dictionary[title])>1:
        counter=1
        for file in df_dictionary[title]:
            print('{}. {}'.format(counter,file))
            counter+=1
        selected_file=int(input('There are {} files for {}. Which file do you want to use? '.format(len(df_dictionary[title]),title)))-1         


    return selected_file

def explode(gdf):
    """ 
    Explodes a geodataframe 
    
    Will explode muti-part geometries into single geometries. Original index is
    stored in column level_0 and zero-based count of geometries per multi-
    geometry is stored in level_1
    
    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries
        
    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with a new index 
                                 and two new columns: level_0 and level_1
        
    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out

def explode_and_rename(df):
    
    df_col_list=list(df)
    df_col_list.remove('geometry')
    df_col_list_x=[col_name+'_x' for col_name in df_col_list]
    df_col_list_y=[col_name+'_y' for col_name in df_col_list]
    col_dict=dict(zip(df_col_list_x,df_col_list))
    df=explode(df)
    df.rename(columns=col_dict,inplace=True)
    df.drop(df_col_list_y,axis=1,inplace=True)
    return df

def intersection_percentage(encompassing_df, subset_df,police_area_col_index):
    col_list=list(encompassing_df)
    #col_number=0
    #for item in col_list:
    #    col_number=col_number+1
    #    print('{}. {}'.format(col_number,item))
    #police_area_col_index=int(input('Which column contains the Police Area Name?'))-1
    data=[]
    for i, subset_df_poly in enumerate(subset_df["geometry"]):
        for j, encompassing_df_poly in enumerate(encompassing_df["geometry"]):
            if subset_df_poly.intersects(encompassing_df_poly):
                data.append([encompassing_df.iloc[j,police_area_col_index], subset_df.iloc[i,2],subset_df.iloc[i,3], round((subset_df_poly.intersection(encompassing_df_poly).area/subset_df_poly.area),5)])
    df = pd.DataFrame(data,columns=['Police_Area_Name','Census','CensusGeoId','Percentage'])
    dissolved_df=encompassing_df.dissolve(by=col_list[police_area_col_index]).reset_index()
    dissolved_df.rename(columns={ dissolved_df.columns[0]: 'Police_Area_Name' }, inplace=True)
    
    return [df, dissolved_df, encompassing_df]

def find_epsg(police_gdf,census_gdf,epsg_df,police_area_col_index,state='none',):
    conversion_epsg=int(census_gdf.crs['init'].split(':')[1])
    epsg_df=epsg_df[epsg_df['PROJ'].str.contains('EPSG',case=False)]
    if state!='none':
        crs_list=list(epsg_df[epsg_df['PROJ Description'].str.contains(state,case=False)]['PROJ'])
    else: crs_list=list(epsg_df['PROJ'])
    crs_list.sort()
    
    continue_key=1
    item_key=0
    print('CRS LIST IS {} items long.'.format(len(crs_list)))
    while item_key<len(crs_list):
        print('{}. {}'.format(item_key+1,crs_list[item_key]))
        temp_df=police_gdf.copy()
        temp_df.crs= {'init' :crs_list[item_key]}
        temp_df=temp_df.to_crs(epsg=conversion_epsg)
        #temp_intersection=intersection_percentage(temp_df, census_gdf)
        if len(gpd.sjoin(temp_df, census_gdf, how="inner", op='intersects'))>0:
            return intersection_percentage(temp_df, census_gdf, police_area_col_index)
        item_key+=1        
    if (state!='none'):
        print('RUNNING FULL LIST')
        return find_epsg(police_gdf,census_gdf,epsg_df)
    else:
        print('Could not find a match')
        return None

def convert_geo_dataframes(police_gdf, census_gdf, epsg_df):
    police_area_col_index=list(police_gdf).index('DISTRICT')
    conversion_epsg=int(census_gdf.crs['init'].split(':')[1])
    if not police_gdf.crs:
        temp_df=police_gdf.copy()
        epsg_input=input('The Police GDF does not have a CRS key. What is the EPSG code to use? ')
        try:
            int(epsg_input)
            
            temp_df.crs= {'init' : 'epsg:'+epsg_input}
            temp_df=temp_df.to_crs(epsg=conversion_epsg)
            if len(gpd.sjoin(temp_df, census_gdf, how="inner", op='intersects'))>0:
                z=intersection_percentage(temp_df, census_gdf, police_area_col_index)
                intersection_df=z[0]
                dissolved_df=z[1]
        except Exception as e:
            print('exception')
            state_input=input('Do you know the state the police departmet is in? (full state name): ')
            if len(list(epsg_df[epsg_df['PROJ Description'].str.contains(state_input,case=False)]['PROJ']))==0:
                state_input='none'
            z=find_epsg(police_gdf,census_gdf,epsg_df,police_area_col_index,state_input, )
            intersection_df=z[0]
            dissolved_df=z[1]
            
    else:
        police_gdf=police_gdf.to_crs(epsg=conversion_epsg)
        
        z=intersection_percentage(police_gdf, census_gdf, police_area_col_index)
        intersection_df=z[0]
        dissolved_df=z[1]
    intersection_df['CensusGeoId']=intersection_df['CensusGeoId'].astype(np.int64)
    return [intersection_df, dissolved_df, z[2]]

def final_acs_data(df_dictionary):
    temp_df=pd.DataFrame()
    for file in df_dictionary['acs_data_to_be_tidied'].keys():
        temp_df=pd.concat([temp_df,massage_acs(df_dictionary['acs_data_to_be_tidied'][file])],sort=True)
    for item in ['Rate', 'Ratio', 'Percent']:
        temp_df['COUNT']=np.where(temp_df['CATEGORY'].str.contains(item),((temp_df['COUNT']/100)*temp_df['TOTAL']),temp_df['COUNT'])
    temp_df['COUNT']=np.where(temp_df['CATEGORY'].str.contains('Median'),((temp_df['COUNT'])*temp_df['TOTAL']),temp_df['COUNT'])
    return temp_df


def incident_df_count(df_dictionary):
    temp_incident_df=df_dictionary['incident_df'].copy()
    if 'SUBJECT_RACE' in list(temp_incident_df):
        search_list=['b','w','a','h','l']
        for item in search_list:
            temp_incident_df['SUBJECT_RACE']=np.where(temp_incident_df['SUBJECT_RACE'].str.lower().str.startswith(item),'INCIDENT_'+item.upper(),temp_incident_df['SUBJECT_RACE'])
        temp_incident_df['SUBJECT_RACE']=np.where(~temp_incident_df['SUBJECT_RACE'].str.lower().str.contains('|'.join(search_list)),'INCIDENT_Other/Unk',temp_incident_df['SUBJECT_RACE'])

        temp_incident_df=temp_incident_df.groupby(['DISTRICT','SUBJECT_RACE']).count().reset_index()
        temp_incident_df=temp_incident_df[temp_incident_df.columns[:3]]
        temp_col_to_rename=list(temp_incident_df)[-1]
        aggregate_df=temp_incident_df.groupby('DISTRICT').sum().reset_index()

        aggregate_df.rename(columns={temp_col_to_rename:'TOTAL'}, inplace=True)
        temp_incident_df=pd.merge(temp_incident_df,aggregate_df, how='left', left_on='DISTRICT', right_on='DISTRICT')
        aggregate_df['SUBJECT_RACE']='INCIDENT_TOTAL'
        temp_incident_df=pd.concat([temp_incident_df,aggregate_df],sort=True)
        temp_incident_df.rename(columns={'DISTRICT':'Police_Area_Name', 'SUBJECT_RACE':'CATEGORY', temp_col_to_rename:'COUNT'},inplace=True)
        temp_incident_df['RATIO']=temp_incident_df['COUNT']/temp_incident_df['TOTAL']*100
        temp_incident_df['COUNT'].fillna(temp_incident_df['TOTAL'], inplace=True)
        temp_incident_df['RATIO'].fillna(temp_incident_df['TOTAL'], inplace=True)
    else:
        temp_incident_df=temp_incident_df.groupby('DISTRICT').count().reset_index()
        temp_col_to_rename=list(temp_incident_df)[-1]
        temp_incident_df.rename(columns={'DISTRICT':'Police_Area_Name', temp_col_to_rename:'COUNT'},inplace=True)
        temp_incident_df['TOTAL']=temp_incident_df['COUNT']
        temp_incident_df['RATIO']=temp_incident_df['COUNT']/temp_incident_df['TOTAL']*100
        temp_incident_df['COUNT'].fillna(temp_incident_df['TOTAL'], inplace=True)
        temp_incident_df['RATIO'].fillna(temp_incident_df['TOTAL'], inplace=True)
    return temp_incident_df
    
def create_and_draw_fig(df, overarching_title, first_map_title, second_map_title, third_map_title, bubble_graph_title, first_map_col, second_map_col, third_map_col, bubble_size, bubble_color):
    fig = tools.make_subplots(rows=2, cols=3, specs=[[{}, {}, {}],[{'colspan': 2},None,None]],
                              subplot_titles=(first_map_title, second_map_title, third_map_title,bubble_graph_title),
                              horizontal_spacing = 0.1, vertical_spacing = 0.05)
    fig=draw_map(df,first_map_col,fig,1,1)
    fig=draw_map(df,second_map_col,fig,1,2)
    fig=draw_map(df,third_map_col,fig,1,3)
    fig=draw_scatter(df,first_map_col,second_map_col, bubble_size, fig, 2,1,bubble_color)
    fig['layout'].update(showlegend=False, height=800, width=800, title=overarching_title)
    for item in ['xaxis1','yaxis1','xaxis2','yaxis2', 'xaxis3', 'yaxis3']:
        fig['layout'][item].update(showgrid=False, mirror=True,showline=True, showticklabels=False)
    fig['layout']['xaxis4'].update(range=[0, 100], showgrid=True, title=second_map_col)
    fig['layout']['yaxis4'].update(range=[0, 100], showgrid=True, title=first_map_col)


    iplot(fig,filename='x')
    return None

def create_and_draw_fig2(df, overarching_title, second_map_title, second_map_col, ):
    fig = tools.make_subplots(rows=2, cols=3, specs=[[{}, {}, {}],[{'colspan': 2},None,None]],
                              subplot_titles=('', second_map_title, '', ''),
                              horizontal_spacing = 0.1, vertical_spacing = 0.05)
    
    fig=draw_map(df,second_map_col,fig,1,2)
    fig['layout'].update(showlegend=False, height=800, width=800, title=overarching_title)
    for item in ['xaxis1','yaxis1','xaxis2','yaxis2', 'xaxis3', 'yaxis3']:
        fig['layout'][item].update(showgrid=False, mirror=True,showline=True, showticklabels=False)
    

    iplot(fig,filename='x')
    return None
def pol_rename_col(state_cd, city_cd, police_gdf):
    if (state_cd=='37') & (city_cd==49):
        return 'Name'
    if (state_cd=='49') & (city_cd==35):
        return 'pol_dist'
    if (state_cd=='35') & (city_cd==16):
        if 'Sector' in list(police_gdf):
            return 'Sector'
        elif 'Division' in list(police_gdf):
            return 'Division'
        elif 'District' in list(police_gdf):
            return 'District'
        else:
            return None
        
def draw_final_map(race,final_df, dissolved_df):
    incident_race=list(race)[0].upper()
    temp=final_df[final_df['CATEGORY'].isin(['INCIDENT_' + incident_race,'INCIDENT_TOTAL', 'RACE - One race - ' + race, 'RACE'])]
    temp=temp[['Police_Area_Name', 'CATEGORY','COUNT']].pivot_table(values='COUNT', index='Police_Area_Name', columns='CATEGORY').reset_index()
    if 'INCIDENT_' + incident_race in list(final_df['CATEGORY']):
        temp[incident_race + '_INCIDENT_RATE']=round(temp['INCIDENT_' + incident_race]/temp['INCIDENT_TOTAL']*100,2)
        temp[incident_race + '_POPULATION_RATIO']=round(temp['RACE - One race - ' + race]/temp['RACE']*100,2)
        temp[incident_race + '_POPULATION_VS_INCIDENT_RATE']=round(temp['INCIDENT_' + incident_race]/temp['RACE - One race - ' + race]*100,2)
        temp=temp[['Police_Area_Name',
                   'INCIDENT_TOTAL',
                   incident_race + '_INCIDENT_RATE',
                   incident_race + '_POPULATION_RATIO',
                   incident_race + '_POPULATION_VS_INCIDENT_RATE']]
        dissolved_df=explode_and_rename(dissolved_df)
        final=pd.merge(temp,dissolved_df[['Police_Area_Name','geometry']], how='right')
        final.fillna(0,inplace=True)
        create_and_draw_fig(final, 
                        'GRAPH AND BUBBLE CHART', 
                        incident_race + ' INCIDENT RATE',
                        incident_race + ' POPULATION RATE', 
                        'TOTAL INCIDENT COUNT',
                        'BUBBLE', 
                        incident_race + '_INCIDENT_RATE', 
                        incident_race + '_POPULATION_RATIO',
                        'INCIDENT_TOTAL',
                        'INCIDENT_TOTAL', 
                        'blue')






    else:
        #temp[incident_race + '_INCIDENT_RATE']=round(temp['INCIDENT_' + incident_race]/temp['INCIDENT_TOTAL']*100,2)
        temp[incident_race + '_POPULATION_RATIO']=round(temp['RACE - One race - ' + race]/temp['RACE']*100,2)
        #temp[incident_race + '_POPULATION_VS_INCIDENT_RATE']=round(temp['INCIDENT_' + incident_race]/temp['RACE - One race - ' + race]*100,2)
        temp=temp[['Police_Area_Name',
                   #'INCIDENT_TOTAL',
                   #incident_race + '_INCIDENT_RATE',
                   incident_race + '_POPULATION_RATIO',
                   #incident_race + '_POPULATION_VS_INCIDENT_RATE']]
                   ]]
        dissolved_df=explode_and_rename(dissolved_df)
        final=pd.merge(temp,dissolved_df[['Police_Area_Name','geometry']], how='right')
        final.fillna(0,inplace=True)
        create_and_draw_fig2(final, 
                        'GRAPH AND BUBBLE CHART', 

                        incident_race + ' POPULATION RATE',
                        incident_race + '_POPULATION_RATIO')
    return None
path="../input/data-science-for-good/cpe-data/"
census_path="../input/census-tract-"
def start_CPE(file,race):
    dept_path=ask_for_dept(path,file)

    df_dictionary=check_departments_sub_folders(dept_path)
    df_dictionary=clean_file_dictionary(path,df_dictionary)
    state_cd=get_state_cd(dept_path)[0]
    city_cd=get_state_cd(dept_path)[1]
    dept='Dept_'+str(state_cd).zfill(2)+'-'+str(city_cd).zfill(5)
    df_dictionary=check_for_census_shapefile(path,state_cd,city_cd,df_dictionary,census_path)
    df_dictionary=load_dataframes(df_dictionary)
    if 'SUBJECT_RACT' in list(df_dictionary['incident_df']):
        df_dictionary['incident_df'].rename(columns={'SUBJECT_RACT':'SUBJECT_RACE'},inplace=True)
    pol_rename=pol_rename_col(state_cd, city_cd, df_dictionary['police_gdf'])
    df_dictionary=rename_district_columns(df_dictionary,pol_rename)
    df_dictionary['crs_reference_df']=pd.read_csv("../input/crs-reference/crs_references.csv")
    percentage_and_dissolved_df_list=convert_geo_dataframes(df_dictionary['police_gdf'], df_dictionary['census_gdf'], df_dictionary['crs_reference_df'])
    percentage_df=percentage_and_dissolved_df_list[0]
    dissolved_df=percentage_and_dissolved_df_list[1]
    df_dictionary['police_gdf']=percentage_and_dissolved_df_list[2]
    working_df1=final_acs_data(df_dictionary)
    working_df=pd.merge(working_df1,percentage_df[['CensusGeoId','Police_Area_Name','Percentage']], how='right', left_on='CENSUS_TRACT',right_on='CensusGeoId')
    if len(working_df)==0:
        working_df=working_df1.copy()
        working_df['Percentage']=1
    working_df['COUNT']=working_df['COUNT']*working_df['Percentage']
    working_df['TOTAL']=working_df['TOTAL']*working_df['Percentage']
    working_df.drop_duplicates(inplace=True)
    working_df=working_df.groupby(['Police_Area_Name','CATEGORY']).sum().reset_index()

    working_df=working_df[['Police_Area_Name', 'CATEGORY', 'COUNT', 'TOTAL']]
    working_df['RATIO']=working_df['COUNT']/working_df['TOTAL']
    incident_df=incident_df_count(df_dictionary)
    final_df=pd.concat([incident_df, working_df], sort=True)
    draw_final_map(race, final_df, dissolved_df)
    return None
start_CPE('Dept_11-00091','White')
start_CPE('Dept_11-00091','Black or African American')
start_CPE('Dept_23-00089','White')
start_CPE('Dept_23-00089','Black or African American')
start_CPE('Dept_23-00089','White')
start_CPE('Dept_23-00089','Black or African American')