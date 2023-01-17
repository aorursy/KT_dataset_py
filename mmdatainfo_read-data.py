import pandas as pd

import numpy as np

# import mysql.connector as sqlcon
file_in = "../input/test-data/movies.csv";

movies = pd.read_csv(file_in,

            parse_dates=["Released"], # parse date

            dtype={"imdbRating":np.float64, # explicit type declaration

                   "tomatoRating":np.int64})
movies.dtypes
movies.head(3)
file_in = "../input/test-data/price_minute_value_bitstamp.txt";

bitcoin = pd.read_csv(file_in,

            delimiter="\t", # tab delimter 

            parse_dates=True, # parse date

            index_col="Timestamp", # set timestamp as index

            na_values="—") # parse "-" as np.NaN
bitcoin.dtypes
bitcoin.head(3)
file_in = "../input/test-data/tsf_data.tsf"
# count number of header lines (until "[DATA]")

count_header = 0;

with open(file_in, "r") as fid:

    for line in fid:

        count_header += 1;

        if "[DATA" in line:

            break;
tsf = pd.read_csv(file_in,

                 delimiter="  ",# double space

                 engine = "python", # use python engine to allow for double space delimter

                 skipinitialspace = True, # if more than two spaces use as delimiter

                 skiprows = count_header, # use found number of header lines

                 header = None, # do not try to read header names (not present)

                 names = ["data","time","grav","pres"],

                 parse_dates = {"datetime":[0,1]},

                 )     
tsf.head()
file_in = "../input/test-data/movies.xlsx";

movies = pd.read_excel(file_in,

            parse_dates=["Released"], # parse date

            dtype={"imdbRating":np.float64, # explicit type declaration

                   "tomatoRating":np.int64})
movies.head(3)
movies.dtypes
# # Connect to the database. Here direct credentials are used. Prefer credentials in separate file

# cnx = sqlcon.connect(user='testuser', password='testpassword',

#                     host='localhost',

#                     database='hosgo')
# # SELECT statement using a variable in the sqlquery

# sqlcursor = cnx.cursor()

# sqlquery = "SELECT * FROM categories WHERE units = \"{}\"".format("W.m^-2");

# sqlcursor.execute(sqlquery)
# # Convert to DataFrame

# pd.DataFrame(sqlcursor.fetchall(),columns=sqlcursor.column_names)
# # Close connection

# sqlcursor.close()

# cnx.close()