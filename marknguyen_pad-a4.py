import pandas as pd
# Import semi clean walmart data
# Hint: The name of the file is located in the Workspace section in the right navigation bar in the Kaggle interface.

# Check the data types for the imported data frame

# Lower case all column values
# Rename IsHoliday column to 'holiday': Refer to the documentation on renaming columns: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html

# Convert Yes and No values to True/False Boolean values

# Convert Dept to integer

# Convert weekly_sales to float:
# Hint: Before converting to float, make sure there are only numbers that are in the string!


# Convert date to actual datetime.datetime objects
# Hint: Use the helper function in Walkthrough 4-2 that was provided to you


# Make sure all values in date column conform to what the helper function expects

# Label the rows with date value of 0 to be "Invalid Date"

# Save the Invalid Date rows to a separate dataframe for future analysis. Make to sure to copy the DataFrame to prevent future reference issues


# Get rid of the Invalid Date rows in the clean_df DataFrame. Make to sure to copy the DataFrame to prevent future reference issues

# Convert the dates - This may take around ~10 seconds

# Confirm that all data has been cleaned and converted to the proper date types
display(clean_df.info()) # display function allows you to display multiple pretty outputs in one cell
display(clean_df.head())