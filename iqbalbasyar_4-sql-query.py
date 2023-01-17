import sqlite3
import pandas as pd

conn = sqlite3.connect("/kaggle/input/algoritma-data-analysis/chinook.db")

albums = pd.read_sql_query("SELECT * FROM albums", conn)
albums.head()
import sqlite3
import pandas as pd

conn = sqlite3.connect("kaggle/input/algoritma-data-analysis/chinook.db")

albums = pd.read_sql_query("SELECT * FROM employees", conn)
albums.head()
## Your code below


## -- Solution code
pd.read_sql_query("SELECT * FROM artists LIMIT 5", 
                  conn, 
                  index_col='ArtistId')
albums = pd.read_sql_query("SELECT AlbumId, Title, a.Name \
                           FROM albums \
                           LEFT JOIN artists as a \
                           ON a.ArtistId = albums.ArtistId", conn)
albums.head()
pd.read_sql_query("SELECT * FROM albums", conn).head()
## Your code below


## -- Solution code
top_cust = pd.read_sql_query("SELECT CustomerId, SUM(Total)  as TotalValue, \
                              COUNT(InvoiceId) as Purchases \
                              FROM INVOICES \
                              GROUP BY CustomerId \
                              ORDER BY TotalValue DESC \
                              LIMIT 5", con=conn, index_col='CustomerId')
top_cust
## Your code below


germany = pd.read_sql_query("SELECT * FROM invoices WHERE BillingCountry = 'Germany'", conn)
germany.head()
not_germany = pd.read_sql_query("SELECT * FROM invoices WHERE BillingCountry IS NOT 'Germany'", conn)
not_germany.head()
america = pd.read_sql_query("SELECT * FROM invoices WHERE BillingCountry IN ('USA', 'Canada')", conn)
america.head()
## Your code below


## -- Solution code
germany.dtypes
invoices_table = pd.read_sql_query("SELECT sql FROM sqlite_master \
                                    WHERE name = 'invoices'", conn)
print(invoices_table.loc[0,:].values[0])
germany_2012 = pd.read_sql_query("SELECT * FROM invoices \
                                  WHERE InvoiceDate >= '2012-01-01' AND InvoiceDate <= '2012-12-31'",
                                 con=conn, parse_dates='InvoiceDate')
germany_2012['InvoiceDate'].describe()
# Your code below

germany = pd.read_sql_query("SELECT * FROM invoices \
                             WHERE BillingCountry = 'Germany' \
                             AND BillingPostalCode LIKE '107%'", conn)
germany.head()
customerinv = pd.read_sql_query("SELECT firstname, lastname, email, company, \
                                 invoiceid, invoicedate, billingcountry, total \
                                 FROM invoices \
                                 left join customers \
                                 on invoices.customerId = customers.customerId", conn)
customerinv.head()
## Your code below


## -- Solution code
customerinv = pd.read_sql_query("SELECT invoices.*  \
                                 FROM invoices \
                                 WHERE invoices.CustomerId IN ( \
                                 SELECT c.CustomerId FROM Customers as c \
                                 LEFT JOIN invoices as i on i.CustomerId = c.CustomerId \
                                 GROUP BY c.CustomerId \
                                 ORDER BY SUM(Total) DESC LIMIT 10)", conn)
customerinv.head()
customerinv = pd.read_sql_query("SELECT *  \
                                 FROM invoices \
                                 WHERE InvoiceId IN (46, 175, 198)", conn)
customerinv.head()
## Your code below


## -- Solution code
## Your code below


## Your code below


## -- Solution code