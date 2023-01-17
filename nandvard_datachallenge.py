import pandas, pandasql, qgrid, plotly, plotly_express

# Read Data
zillow = pandas.read_csv('../input/Zip_Zhvi_2bedroom.csv')
airbnb = pandas.read_csv('../input/listings.csv',low_memory=False)

# Clean & Format datatypes
zillow['RegionName'] = zillow['RegionName'].astype(str)
zillow['2017-06'] = zillow['2017-06'].astype(int)
airbnb['price'] = airbnb['price'].str.replace('\$|,|\.00','').astype(int)
airbnb['last_review'] = pandas.to_datetime(airbnb['last_review'])

# SQL Query
query = """
SELECT
    zipcode,
    neighbourhood_group_cleansed AS Area,

    count(price) AS Properties,  -- no. of properties, to find popular areas
    sum(number_of_reviews) AS Reviews,  -- no. of reviews, to find popular properties
    
    [2017-06] AS [Cost$],  -- use 2017 cost
    (cast(AVG(price) * 365 * .75 as int))/100*100 AS [Revenue$/Year],  -- use daily price and 75% occupancy
    
    round([2017-06]/(AVG(price) * 365 * .75),2) AS BreakEven_Years
FROM
    airbnb a
JOIN
    zillow z ON a.zipcode = z.RegionName
WHERE 1=1
    AND a.bedrooms = 2 AND z.city = 'New York'  -- filter client requirements

    AND a.number_of_reviews > 2 AND a.last_review > date('now','-2 years')  -- filter popular properties
GROUP BY
    zipcode
HAVING
    Properties > 2  -- filter popular areas
ORDER BY
    BreakEven_Years
"""

az = pandasql.sqldf(query)

# Interactive Grid
display(qgrid.show_grid(az)) 


plotly.offline.init_notebook_mode()
# ROI Quadrant
fig = plotly_express.scatter(az, x="Revenue$/Year", y="BreakEven_Years", color="Properties", size="Cost$",
                             hover_data=['Cost$','Area','zipcode'],
                             title='ROI Quadrant : Bottom-Right - High Revenue & Quick BreakEven')
fig.show()
