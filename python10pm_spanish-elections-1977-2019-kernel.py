import pandas as pd

pd.set_option("max_columns", None)



# show several prints on the same cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



def zip_rows(df):

    '''

    Expects 2 rows, and zips the names to generate the header column to our df.

    '''

    

    row1 = list(df.columns)

    row2 = list(df.loc[0])

    col_names = []

    

    for d1, d2 in zip(row1, row2):

        if d2.lower() == "diputados":

            pass

        elif "unnamed" in d1.lower() and d2.lower() != "votos":

            col_names.append(d2.lower())

        else:

            d1_append = d1 + " diputados"

            d2_append = d1 + " " + d2

            col_names.append(d1_append)

            col_names.append(d2_append)



    return col_names



def get_excel_columns(file):

    '''

    Get's the Excel column names.

    Credit: https://github.com/pandas-dev/pandas/issues/16645

    ---------------------------------------------------------

    We will use it to create dynamic names because each party

    has 2 columns: # votes and # deputy elected

    '''

    

    extension = os.path.splitext(file)[1]

    assert extension == ".xlsx", "The file has to be and Excel spreadsheet."

    

    # create an excel file object

    workbook = pd.ExcelFile(file)



    # get the total number of rows (assuming you're dealing with the first sheet)

    rows = workbook.book.sheet_by_index(0).nrows

    

    # skip 4 rows, since the data has some formatting rows with no data value

    skiprows = 4



    # get the 2 rows that we need

    workbook_dataframe = pd.read_excel(workbook, skiprows=skiprows, skip_footer=(rows-6))

    

    col_names = zip_rows(workbook_dataframe)

    

    return col_names



for dirname, _, filenames in os.walk('/kaggle/input'):

    for f in filenames:

        print(f)

        extension = os.path.splitext(f)[1]

        if extension == ".xlsx":

            try:

                f = os.path.join(dirname, f)

                col_names = get_excel_columns(f)

                df = pd.read_excel(f,  skiprows=5, names=col_names)

                df.head()

            except:

                print(f)

                print("Something went wrong!")

       
#PS: I personally had a lot of fun working with the datasets. Especially the cleaning part of the DF, since they don't come

# in a normal CSV format.