# This is super ugly at the moment - Mostly trying to get a handle on creating a 
# sustainable way to convert to GPS points, and find combined area of precincts / census areas

# I'm so new to Kaggle that I don't know how to make it easy to access the files in the 
# .zip archives I included - apologies!
# I added some new State Census info, but am so new to Kaggle that I don't 
# know how to make it easily accessible after uploading the zip file containing it

# Additinoally, the APD_DST.prj file goes with Dept_37-00027 (thanks to Chris)
# (https://www.kaggle.com/crawford/another-world-famous-starter-kernel-by-chris)
# for the info required for the .prj file 
# (though I found I had to take out the 'AUTHORITY["EPSG","102739"]' at the end of the .prj
# file for it to work for me)
import numpy as np
import matplotlib.pyplot as plt
import fiona
import pandas as pd
from pyproj import Proj, transform
from shapely.geometry import Polygon
from fiona.crs import to_string

class Census:
    def __init__(self, census_path):
        self.census_path = census_path


    def get_coords(self):
        census_shapefile = fiona.open(self.census_path)
        census_boundaries = {}
        for row in census_shapefile:
            geoid = row['properties']['GEOID']
            coordinates = row['geometry']['coordinates'][0]
            census_boundaries[geoid] = coordinates
        return census_boundaries

class Department:
    def __init__(self, department_shapefile_path):
        self.shapefile = fiona.open(department_shapefile_path)
        self.crs_string = to_string(self.shapefile.crs)

    def convert_department_shapefile_to_gps_coords(self):
        all_departments = {}
        for rec in self.shapefile:
            coords = rec['geometry']['coordinates']
            for area in coords:
                department_polygon = []
                for coord in area:
                    myproj = Proj(self.crs_string, preserve_units=True)
                    newcoord = myproj(coord[0], coord[1], inverse=True)
                    department_polygon.append(newcoord)
                all_departments[rec['id']] = department_polygon
        return all_departments

    def convert_star_stle(self,star, stle):
        myproj = Proj(self.crs_string, preserve_units=True)
        newcoord = myproj(star, stle, inverse=True)
        return newcoord

class Incidents:
    def __init__(self, incidents_path):
        df = pd.read_csv(incidents_path)
        self.df = df[1:]

    def load_dataframe(self):
        return self.df

    def get_dataframe(self):
        self.get_datetime_index()
        self.remove_no_gps()
        return self.df

    def get_datetime_index(self):
        combined_datetime = self.get_combined_datetime()
        self.df['DATETIME'] = pd.to_datetime(combined_datetime)
        self.df.drop(['INCIDENT_DATE'], axis=1,inplace=True)

    def get_combined_datetime(self):
        combined_datetime = self.df['INCIDENT_DATE']
        keys = self.df.keys()
        if 'INCIDENT_TIME' in keys:
            combined_datetime = self.df['INCIDENT_DATE'].str.cat(df['INCIDENT_TIME'], sep=' ')
            self.df.drop(['INCIDENT_TIME'],axis=1,inplace=True)

        return combined_datetime
    def remove_no_gps(self):
        return self.df.dropna(subset=['LOCATION_LATITUDE', 'LOCATION_LONGITUDE'])


class Overlap:
    def percentage_of_county_in_precinct(self, county, precinct):
        final_array = []
        for county_id, county_coords in county.items():
            if len(county_coords) < 3:
                continue
            for precinct_id, precinct_coords in precinct.items():
                if len(precinct_coords) < 3:
                    continue
                row_array = {}
                precinct_polygon = Polygon(precinct_coords)
                county_polygon = Polygon(county_coords)
                overlap = county_polygon.intersection(precinct_polygon)
                overlap_area = (overlap.area / county_polygon.area) * 100
                row_array['county_id'] = county_id
                row_array['precinct_id'] = precinct_id
                row_array['overlap_area'] = overlap_area
                final_array.append(row_array)

        return final_array

state_shapefile_path = "state-data/texas/cb_2017_48_tract_500k.shp"
department_shapefile_path = "provided-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"
incidents_path = "provided-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv"

department = Department(department_shapefile_path)
department_coords = department.convert_department_shapefile_to_gps_coords()
print("Loaded departments")
incidents = Incidents(incidents_path)
df = incidents.get_dataframe()
print("Loaded Incidents")
census = Census(state_shapefile_path)
census_coords = census.get_coords()
print("Loaded Census")
overlap = Overlap()
overlap_percentage = overlap.percentage_of_county_in_precinct(census_coords,department_coords)
print("Loaded overlap")
overlap_df = pd.DataFrame(overlap_percentage)
print(overlap_df[(overlap_df['overlap_area'] > 0)].head(100))


# Print a quick map of the areas in the shapefile to sanity check
import shapefile as shp
import matplotlib.pyplot as plt

department_shapefile_path = "provided-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"
sf = shp.Reader(department_shapefile_path)

plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
plt.show()
# TO DO:
# 1. From GPS coordinate of incident, determine which precinct it occured in
# 2. From GPS coordinate of incident, determine which Census tract it occured in
# 3. Attach Precinct ID and Census ID to each incident type
# 4. Determine overlap % of Census Tracts with Precincts (separate table / dataframe?  Pivot table, at any rate)
# 5. For each precinct, count total # of white, black, and other minority residents
# 6. Create function to determine if coordinates need to be converted