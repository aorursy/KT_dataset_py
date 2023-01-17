# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import geopandas as gpd
import geopy as gpy
from geopandas import GeoDataFrame
from shapely.geometry import Point
# Set Google Maps API key for use later on - this is required to run the find_crs() function below
# Register for free at https://cloud.google.com/maps-platform/
read_in = pd.read_csv("../input/api-key/key.csv")
api_key = read_in["Key"][0]
# Read in the Texas census tract file - a separate census tract zip file is required for each state
# In this example I'll be sticking to Texas!
texas_tracts = gpd.read_file("../input/census-tracts-texas-censusgov/cb_2017_48_tract_500k/cb_2017_48_tract_500k.shp")

# Read in FIPS County Codes for lookup reference
FIPS = pd.read_csv("../input/cpe-helper-data/FIPS_County_Codes.csv", dtype = "object")

# Read in the US Postal Service State Codes
usps_df = pd.read_csv("../input/cpe-helper-data/USPS_State_Codes.csv")

# Read in the coordinate reference systems compiled via spatial reference.
# Note, I selected 823 of the "most likely to be used" - this list could be expanded for thoroughness!
projection_df = pd.read_csv("../input/cpe-helper-data/CRS_References.csv")
# Read in initial list of departments available (the top level of our directory structure)
dept_list = [d for d in next(os.walk("../input/data-science-for-good/cpe-data"))[1] if not d.startswith((".", "_"))]

# Read in directories per department (the next level down)
dept_dir_dict = {}
for dept in dept_list:
    depts = [d for d in next(os.walk(os.path.join("../input/data-science-for-good/cpe-data", dept)))[1] if not d.startswith((".", "_"))]
    dept_dir_dict.update({dept:depts})
    
# Read in sub-directories per directory (one more level down)
dept_subdir_dict = {}
for dept in dept_dir_dict:
    for subdir in dept_dir_dict[dept]:
        subsubdir = [d for d in next(os.walk(os.path.join("../input/data-science-for-good/cpe-data", dept, subdir)))[1] if not d.startswith((".", "_"))]
        dept_subdir_dict.update({subdir:subsubdir})
        
# For each department we expect ACS data - get a list of ACS directories
sub_dir_acs = []
for i in range(len(dept_list)):
    sub = [s for s in dept_dir_dict[dept_list[i]] if "ACS" in s][0]
    sub_dir_acs.append(sub)
    
# Within each ACS directory we expect 5 sub-folders for each category of data - get a list of ACS sub-directories
sub_dir_acs_det = []
for i in range(len(sub_dir_acs)):
    subsub = [s for s in dept_subdir_dict[sub_dir_acs[i]]]
    sub_dir_acs_det.append(subsub)
for i in range(len(sub_dir_acs_det)):
    sub_dir_acs_det[i].sort()
    
# Create a dictionary that can be used to reference which type of ACS data we want to retrieve
acs_dict = {"education" : 0, "education25" : 1, 
           "housing" : 2, "poverty" : 3, "rsa" : 4}

# And create a dictionary to go back - from dictionary number to ACS descriptions
inv_acs_dict = {v: k for k, v in acs_dict.items()}
    
# For each department we expect shapefile data - get a list of shapefile directories
sub_dir_shp = []
for i in range(len(dept_list)):
    sub = [s for s in dept_dir_dict[dept_list[i]] if "hape" in s][0]
    sub_dir_shp.append(sub)
def read_uoffile(dept):
    """This function reads in UOF data for the requested department.
    Returns a Pandas dataframe after reading in the relevant .csv file."""
    path = os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                                               dept_list[int(dept)])) if "UOF" in f][0]
    full_path = os.path.join(path, file)
    
    df = pd.read_csv(full_path)
    return df

def read_shapefile(dept):
    """This function reads in police shapefile data for the requested department.
    Returns a GeoPandas dataframe after reading in the relevant "shp", "shx", "dbf" and "prj" files."""
    path = os.path.join("../input/data-science-for-good/cpe-data", dept_list[int(dept)], 
                        sub_dir_shp[int(dept)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                                               dept_list[int(dept)], 
                                               sub_dir_shp[int(dept)])) if ".shp" in f][0]
    full_path = os.path.join(path, file)
    
    gdf = gpd.read_file(full_path)
    return gdf

def read_acsfile_key(dept,category):
    """This function reads in ACS data for the requested department and category.
    Returns a FIPS key (where digits 0-1 = State, digits 2-4 = County)."""
    path = os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])) if "ann" in f][0]
    full_path = os.path.join(path, file)
    
    df = pd.read_csv(full_path).head()
    FIPS_info = df.iloc[1, 1]
    return FIPS_info

def read_acsfile(dept,category):
    """This function reads in ACS data for the requested department and category.
    Returns a Pandas dataframe after reading in the relevant .csv file."""
    path = os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])) if "ann" in f][0]
    full_path = os.path.join(path, file)
    
    df = pd.read_csv(full_path)
    return df
def check_shapefiles():
    """This function checks availability of required mandatory shapefiles by department.
    Returns a list of lists containing shapefile extensions."""
    mandatory_files = ["shp", "shx", "dbf", "prj"]
    shapefile_check = []

    for i in range(len(dept_list)):
        row_check = []
        for file in mandatory_files:
            try:
                if [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", dept_list[i], sub_dir_shp[i])) if file in f][0]:
                    row_check.append(file)
            except:
                pass
        shapefile_check.append(row_check)
    return shapefile_check

def check_uoffiles():
    """This function checks availability of required use of force files by department.
    Returns a list UOF if available or None if not available."""
    uof_var = "UOF"
    uoffile_check = []

    for i in range(len(dept_list)):
        try:
            if [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", dept_list[i])) if uof_var in f][0]:
                uoffile_check.append(uof_var)
        except:
            uoffile_check.append("None")
    return uoffile_check

def check_acsfiles():
    """This function retrieves the FIPS codes per ACS file by department.
    Returns a list of lists containing FIPS data.
    """
    FIPS_grid = []
    for j in range(5):
        cat_check = []
        for i in range(len(dept_list)):
            FIPS_info = read_acsfile_key(i,j)
            cat_check.append(FIPS_info)
        FIPS_grid.append(cat_check)
    return FIPS_grid

def color_exceptions(val):
    """Checks for defined exception values and highlights them in red."""
    color = 'red' if (val == "None") or (val == False) or (val == ['shp', 'shx', 'dbf']) else 'black'
    return 'color: %s' % color

def assemble_overview():
    """This function assembles an overview of the data provided and highlights any key issues,
    including 1) missing *.prj files 2) missing use of force files 3) inconsistent ACS data"""
    
    # Let's run the check_shapefiles and get a list of shapefiles available per department
    dept_shapes = check_shapefiles()

    # Let's run the check_uoffiles and get a list of uoffiles available per department
    dept_uofs = check_uoffiles()

    # Let's run the check_acsfiles and get a list of acsfiles available per department
    cat_acs = check_acsfiles()
    
    overview = pd.DataFrame({"dept": dept_list, 
                             "uofs": dept_uofs, 
                             "shapes": dept_shapes, 
                             inv_acs_dict[0] : cat_acs[0],
                             inv_acs_dict[1] : cat_acs[1],
                             inv_acs_dict[2] : cat_acs[2],
                             inv_acs_dict[3] : cat_acs[3],
                             inv_acs_dict[4] : cat_acs[4]})
    overview["state"] = overview["education"].str[0:2]
    overview["education"] = overview["education"].str[0:5]
    overview["education25"] = overview["education25"].str[0:5]
    overview["housing"] = overview["housing"].str[0:5]
    overview["poverty"] = overview["poverty"].str[0:5]
    overview["rsa"] = overview["rsa"].str[0:5]

    # Get county text
    county = []
    for i in overview.index:
        county_val = FIPS.loc[((FIPS["state_code"] == overview.loc[i, "education"][0:2]) & (FIPS["fips_county"] == overview.loc[i, "education"][2:5])), "description"].values[0]
        county.append(county_val)
    overview["county"] = county

    # Get state text keys
    state = []
    for i in overview.index:
        state_val = FIPS.loc[(FIPS["state_code"] == overview.loc[i, "state"]), "state"].values[0]
        state.append(state_val)
    overview["state"] = state

    # Re-assemble the df with fields in required order
    overview = overview[['dept', 'state', 'county', 'uofs', 'shapes', 'education', 'education25', 'housing',
           'poverty', 'rsa']]

    # Add a column reflecting if there are any issues with the ACS data
    acs_check = []
    for i in overview.index:
        eval = overview.loc[i, "education"] == overview.loc[i, "education25"] == overview.loc[i, "housing"] == overview.loc[i, "poverty"] == overview.loc[i, "rsa"]
        acs_check.append(eval)
    overview["acs_ok"] = acs_check

    # And finally, let's color any exceptions found and present our overview
    data_overview = overview.style.applymap(color_exceptions)
    
    return data_overview
overview = assemble_overview()
overview
# Note that it is not currently possible to run this function on Kaggle currently due 
# to the limitation around rtree https://github.com/Kaggle/docker-python/issues/108
def make_geodf_uof_acs(uoffile, long_col, lat_col, acsfile):
    """Converts the specified uoffile with longitude and latitude to a GeoPandas dataframe,
    and ensures that only UOF data within the supplied ACS file boundaries is displayed."""
    geometry = [Point(xy) for xy in zip(uoffile[long_col].astype("float"), 
                                    uoffile[lat_col].astype("float"))]
    uoffile = GeoDataFrame(uoffile, crs = "epsg:4269", geometry = geometry)
    uoffile = gpd.sjoin(uoffile, acsfile, how="inner", op='intersects')
    return uoffile

def make_geodf_uof(df, x_col, y_col):
    """Converts a pandas df to a GeoPandas df, using the specified x and y data to create 'geometry'."""
    geometry = [Point(xy) for xy in zip(df[x_col].astype("float"), 
                                    df[y_col].astype("float"))]
    df = GeoDataFrame(df, geometry = geometry)
    return df

def check_crs(shapefile):
    """The ACS data uses epsg:4269 as a standard, so our aim is to standardize all map co-ordinates on epsg:4269
    so that no matter where the data comes from it can be analysed and plotted within the same projection.
    Returns an evaluation of the relevant shapefile with recommendation where required."""
    target = 'epsg:4269'
    if shapefile.crs == {}:
        print("no initial projection - use find_crs() to find the correct projection") # see later in notebook
    elif shapefile.crs == target:
        print("no problem - ready for mapping")
    else:
        print("requires conversion - use conv_crs() to fix")
        
def conv_crs(shapefile):
    """Converts the specified shapefile from one CRS to our standard epsg:4269"""
    shapefile = shapefile.to_crs(epsg='4269')
    return shapefile
# Read in the ACS file for Dallas (3), Poverty (3) as per overview table above
dallas_acs = read_acsfile(3, 3)

# Rename the GEO.id column to AFFGEOID so it matches to the corresponding tract file column name
dallas_acs.rename(columns = {"GEO.id": "AFFGEOID"}, inplace = True)

# Merge the ACS file for Dallas, Poverty with the Texas tracts data
dallas_acs = dallas_acs.merge(texas_tracts, on = "AFFGEOID")

# And then convert the resulting df to a Gdf
dallas_acs = GeoDataFrame(dallas_acs, crs = "epsg:4269", geometry = dallas_acs["geometry"])

# And then let's look at the resulting output
fig, ax = plt.subplots(figsize = (12, 12))
# Plot the ACS data
ax = dallas_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
# Add a title
fig.suptitle('Dallax Texas, Census Tracts', x = 0.5, y = 0.89)
plt.show()
# Let's read in the police shapefile data given
dallas_shapes = read_shapefile(3)
dallas_shapes.head()
# Let's check how the projection of our police shapefile lines up against our agreed standard for ACS
check_crs(dallas_shapes)
# If we print the crs values we can see they are completely different
print(dallas_acs.crs, "vs", dallas_shapes.crs)
# And we can see visually that it's problematic by trying to plot the ACS data and the police shapefile data together - 
# we get a plot with nothing in it!
fig, ax = plt.subplots(figsize = (10, 10))
ax = dallas_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
dallas_shapes.plot(ax = ax, color = "none", edgecolor = "b", linewidth = 1.5)
fig.suptitle('Dallas Texas, Census Tracts with Police Districts', x = 0.5, y = 0.88)
plt.show()
# That's OK though because remember we do have an initial CRS to work with so 
# let's use our function to convert to our standard epsg:4269
dallas_shapes = conv_crs(dallas_shapes)
# And if we plot ACS data and police shapefile data they are now lining up nicely
fig, ax = plt.subplots(figsize = (12, 12))
# Plot the ACS data
ax = dallas_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
# Add the shapefile data
dallas_shapes.plot(ax = ax, color = "none", edgecolor = "b", linewidth = 1.5)
# Add a title
fig.suptitle('Dallas Texas, Census Tracts with Police Districts', x = 0.5, y = 0.88)
plt.show()
# Now let's read in our UOF data for Dallas - in this case we're lucky as the incident co-ordinates are given in 
# latitude and longitude already...
dallas_uof = read_uoffile(3)
dallas_uof.columns
# We need to do a little cleaning of non-null, non-numeric and data type values before proceeding
dallas_uof.dropna(subset = ["LOCATION_LATITUDE"], inplace=True)
dallas_uof.dropna(subset = ["SUBJECT_RACE"], inplace=True)
dallas_uof.drop([0], inplace = True)
dallas_acs["HC02_EST_VC01"] = dallas_acs["HC02_EST_VC01"].astype("float")
# And now let's convert our use of force data to a GeoPandas dataframe
dallas_uof = make_geodf_uof(dallas_uof, "LOCATION_LONGITUDE", "LOCATION_LATITUDE")
# It would be nice to add labels to our data as a finishing touch so we'll get "representative points"
# for each shape which we'll use to plot our labels later on
dallas_shapes['coords'] = dallas_shapes['geometry'].apply(lambda x: x.representative_point().coords[:])
dallas_shapes['coords'] = [coords[0] for coords in dallas_shapes['coords']]
# Let's get some basic statistics for HC02_EST_VC01 ("Below poverty level; Estimate; Population for whom 
# poverty status is determined") - we'll use this to highight rich vs poor levels on our map
dallas_acs["HC02_EST_VC01"].describe()
fig, ax = plt.subplots(figsize = (12, 12))

# Plot the ACS data by poverty level
ax = dallas_acs.plot(ax = ax, color = "c", edgecolor = "darkgrey", linewidth = 0.5)
dallas_acs[dallas_acs["HC02_EST_VC01"] >= dallas_acs["HC02_EST_VC01"].describe()["50%"]].plot(ax = ax, color = "mediumturquoise", edgecolor = "darkgrey", linewidth = 0.5)
dallas_acs[dallas_acs["HC02_EST_VC01"] >= dallas_acs["HC02_EST_VC01"].describe()["75%"]].plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)

# Add the shapefile data
dallas_shapes.plot(ax = ax, color = "none", edgecolor = "b", linewidth = 1.5)

# Add the use of force data
dallas_uof.plot(ax = ax, color = "orangered", alpha = 0.5, markersize = 10)

# Some labels would also be nice!
texts = []
for i in dallas_shapes.index:
    text_item = [dallas_shapes["coords"][i][0] + 0.015, 
                  dallas_shapes["coords"][i][1] + 0.015, 
                  dallas_shapes["Name"][i]]
    texts.append(text_item)
    plt.text(texts[i][0], texts[i][1], texts[i][2], color = "black")

# And finally there are a lot of colours so a key will be useful   
low_line = matplotlib.lines.Line2D([], [], color='c',markersize=120, label='poverty - low')
med_line = matplotlib.lines.Line2D([], [], color='mediumturquoise',markersize=120, label='poverty - medium')
high_line = matplotlib.lines.Line2D([], [], color='paleturquoise', markersize=120, label='poverty - high')
police_line = matplotlib.lines.Line2D([], [], color='b', markersize=120, label='police precincts')
uof_line = matplotlib.lines.Line2D([], [], color='orangered', markersize=120, label='use of force')
handles = [low_line, med_line, high_line, police_line, uof_line]
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels, fontsize = 10, loc='lower right', shadow = True)

# Add a title
fig.suptitle('Dallas Texas, Use of Force by Poverty Levels and Police District', x = 0.5, y = 0.88)
plt.show()
# Read in the ACS file for Travis(4), Poverty (3) as per overview table above
travis_acs = read_acsfile(4, 3)

# Rename the GEO.id column to AFFGEOID so it matches to the corresponding tract file column name
travis_acs.rename(columns = {"GEO.id": "AFFGEOID"}, inplace = True)

# Merge the ACS file for Dallas, Poverty with the Texas tracts data
travis_acs = travis_acs.merge(texas_tracts, on = "AFFGEOID")

# And then convert the resulting df to a Gdf
travis_acs = GeoDataFrame(travis_acs, crs = "epsg:4269", geometry = travis_acs["geometry"])

# And then let's look at the resulting output

fig, ax = plt.subplots(figsize = (12, 12))
ax = travis_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
fig.suptitle('Travis Texas', x = 0.5, y = 0.84)
plt.show()
# Let's read in the police shapefile data given
travis_shapes = read_shapefile(4)
travis_shapes.head()
# Let's check how the projection of our police shapefile lines up against our agreed standard for ACS
check_crs(travis_shapes)
# If we print the crs values we can see that there is simply NO crs data available for Travis ({})
print(travis_acs.crs, "and", travis_shapes.crs)
# Now let's read in our UOF data for Travis - we seem to have 2 Y co-ordinates(!) as well as longitude and latitude -
# yikes!
travis_uof = read_uoffile(4)
travis_uof.columns
# A closer look at the data reveals that latitude and longitude are seldom given so there are 
# too many null values to be useful. The 2 "Y co-ordinates" are actually X and Y, they've just been
# mis-labelled by whatever process they went through previously. We also observe that our X and Y are
# certainly not latitude or longitude so we're going to have to find a suitable projection,
# AND we are given physical address data in this file, and this is going to provide us with the means
# to determine the right projection (LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION)
travis_uof.head(5)
# Let's re-name our columns for ease of use
travis_uof.rename(columns = {"Y_COORDINATE" : "X_COORDINATE", "Y_COORDINATE.1" : "Y_COORDINATE"}, inplace = True)
# And then do a little cleaning of non-null, non-numeric values
travis_uof.dropna(subset = ["X_COORDINATE"], inplace=True)
travis_uof.drop([0], inplace = True)
travis_uof.drop(list(travis_uof[travis_uof["X_COORDINATE"] == "-"].index), inplace = True)
travis_uof.drop([2094], inplace = True)
travis_acs["HC02_EST_VC01"] = travis_acs["HC02_EST_VC01"].astype("float")
travis_uof.head(5)
def find_crs(rand_geodf_source):
    """Selects 3 random addresses from the specified GeoPandas df and retrieves latitude and longitude for them
    via Google maps API (this is then transformed to our standard epsg:4269. Selects a list of 'likely' projections
    - just based on State for now for demonstration purposes - and then tests what happens when we convert from that 
    CRS to our standard CRS. The CRS with the least difference in distance between our data and Google is deemed 
    the closest match and can be used for conversion."""
    # Now let's get corresponding locations for these addresses from Google
    rand_google = gpd.tools.geocode(rand_addresses.values, provider="googlev3", api_key = api_key)
    # And then convert to the standard projection we've decided upon
    rand_google.to_crs(epsg='4269')
    
    # Let's create a df where we'll store our evaluation data
    rand_eval = pd.DataFrame(rand_geodf_source["address"])
    
    # Get a list of projections to try
    projection_tries = projection_df[projection_df["PROJ Description"].str.contains(usps_df.loc[usps_df["USPS"] == rand_geodf_source["state"][0], "State"].values[0])]
    projection_tries.reset_index(drop = True, inplace = True)
        
    for i in range(len(projection_tries)):
        # First we make a copy of our source data to work on
        rand_geodf = rand_geodf_source.copy()

        # Let's now set the crs to the first one we want to try
        rand_geodf.crs = {'init' : projection_tries["PROJ"][i]}
        # And then convert to our standard
        rand_geodf = rand_geodf.to_crs(epsg='4269')

        # And let's store the outcomes of our first test
        rand_eval[projection_tries["PROJ"][i]] = rand_geodf.distance(rand_google)
    
    # Find the mean of the values for each column
    rand_eval.loc['avg'] = rand_eval.mean()
    
    # Find the best fit
    answer = rand_eval.loc['avg'].dropna().sort_values().index[0]
    
    return answer, rand_eval
# Now we can randomly pick 3 addresses we'll use to validate on
rand_addresskeys = list(np.random.randint(1,len(travis_uof), 3))

# We need the full address for best geo-coding results
travis_uof["FULL_ADDRESS"] = travis_uof["LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION"] + \
", " + travis_uof["LOCATION_CITY"] + \
", " + travis_uof["LOCATION_STATE"]

# Now let's assemble the 3 series we'll use to create our test df
rand_addresses = travis_uof.loc[rand_addresskeys, "FULL_ADDRESS"]
rand_state = travis_uof.loc[rand_addresskeys, "LOCATION_STATE"]
rand_x = travis_uof.loc[rand_addresskeys, "X_COORDINATE"]
rand_y = travis_uof.loc[rand_addresskeys, "Y_COORDINATE"]

# And finally create the test df
rand_address_table = pd.DataFrame({"address" : rand_addresses.values, 
                                   "state" : rand_state,
                                   "X_COORDINATE": rand_x.values, 
                                   "Y_COORDINATE": rand_y.values})

# Convert our rand_address_table to a rand_geodf (a GeoPandas df)
rand_geodf_source = make_geodf_uof(rand_address_table, "X_COORDINATE", "Y_COORDINATE")
rand_geodf_source.reset_index(drop = True, inplace = True)
# Run the find_crs function and then display our final answer
answer, rand_eval = find_crs(rand_geodf_source)
answer
# And also have a look at the top 5 options - notice that there are in fact 4 different projections
# that would minimize the difference between our 'ground truth' co-ordinates obtained from Google
# and our new projections based on our chosen CRS
rand_eval.loc["avg"].dropna().sort_values().head()
# Let's now convert our Travis dataframe to a GeoPandas dataframe
travis_uof = make_geodf_uof(travis_uof, "X_COORDINATE", "Y_COORDINATE")
# And then specify the CRS we identified as best fit
travis_uof.crs = {'init' : answer}
travis_shapes.crs = {'init' : answer}
# After which we can convert to our standard
travis_uof = travis_uof.to_crs(epsg='4269')
travis_shapes = travis_shapes.to_crs(epsg='4269')
# It would be nice to add labels to our data as a finishing touch so we'll get "representative points"
# for each shape which we'll use to plot our labels later on
travis_shapes['coords'] = travis_shapes['geometry'].apply(lambda x: x.representative_point().coords[:])
travis_shapes['coords'] = [coords[0] for coords in travis_shapes['coords']]
fig, ax = plt.subplots(figsize = (12, 12))
# Plot the ACS data
ax = travis_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
# Add the shapefile data
travis_shapes.plot(ax = ax, column = "DISTRICT", cmap = "viridis", vmin = 1.8)
# Add the use of force data
travis_uof.plot(ax = ax, color = "orangered", alpha = 0.2, markersize = 10) 
# Provide a legend
uof_line = matplotlib.lines.Line2D([], [], color='orangered', markersize=120, label='use of force')
handles = [uof_line]
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels, fontsize = 10, loc='lower right', shadow = True)
# Add a title
fig.suptitle('Travis Texas, Use of Force by Police District', x = 0.5, y = 0.82)
plt.show()
