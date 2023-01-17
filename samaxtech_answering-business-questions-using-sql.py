import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image
import warnings

%matplotlib inline

warnings.filterwarnings('ignore')
font = {'family' : 'DejaVu Sans',
        'weight' : 'regular',
        'size'   : 19}

plt.rc('font', **font)
#run_query(q): Takes a SQL query as an argument and returns a pandas dataframe by using the connection as a SQLite built-in context manager. 
def run_query(q):
    with sqlite3.connect('../input/chinook-music-store-data/chinook.db') as conn:
        return pd.read_sql_query(q, conn)
    
#run_command(c): Takes a SQL command as an argument and executes it using the sqlite module.
def run_command(c):
    with sqlite3.connect('../input/chinook-music-store-data/chinook.db') as conn:
        conn.isolation_level = None
        conn.execute(c)
    
#show_tables(): calls the run_query() function to return a list of all tables and views in the database.
def show_tables():
    q = '''SELECT
            name,
            type
        FROM sqlite_master
        WHERE type IN ("table","view");
        '''
    return run_query(q)    
    
#Initial state of the database
show_tables()
Image(filename='../input/chinook-music-store-data/schema_diagram.png')
q1 = '''
    WITH 
        genre_track_sold AS
            (
            SELECT 
                g.name genre,
                il.quantity,
                il.invoice_id
            FROM genre g 
            INNER JOIN track t ON g.genre_id = t.genre_id
            INNER JOIN invoice_line il ON t.track_id = il.track_id
            ),
        
        sold_USA AS
            (
            SELECT
                gts.genre,
                gts.quantity,
                c.country
            FROM genre_track_sold gts
            INNER JOIN invoice i ON i.invoice_id = gts.invoice_id
            INNER JOIN customer c ON c.customer_id = i.customer_id
            WHERE country = 'USA'
            )
    
    SELECT 
        genre,
        SUM(quantity) tracks_sold,
        CAST(SUM(quantity) as float) / (SELECT COUNT(*) FROM sold_USA) percentage
    FROM sold_USA
    GROUP BY 1
    ORDER BY 2 DESC 
    LIMIT 10;
    '''

genre_sales_usa = run_query(q1)
run_query(q1)
genre_sales_usa = genre_sales_usa.set_index('genre', drop=True)
ax = genre_sales_usa.plot.barh(xlim=(0, 625), 
                               colormap=plt.cm.Accent, 
                               legend=False,
                               width=1.1,
                               figsize=(20,10)
                              )

for i, label in enumerate(list(genre_sales_usa.index)):
    score = genre_sales_usa.loc[label, "tracks_sold"]
    label = (genre_sales_usa.loc[label, "percentage"] * 100).astype(int).astype(str) + "%"
    plt.annotate(str(label), (score + 8, i - 0.36))
    
ax.set_title('Top 10 Genres in the USA', fontsize=35, y=1.05)
ax.set_xlabel('Tracks Sold', fontsize=24)
ax.tick_params(axis = 'both', labelsize = 20)
plt.ylabel('')
                
plt.tight_layout()
plt.show();
q2 = '''
    WITH sales_per_customer AS
        (
        SELECT 
            i.customer_id,
            c.support_rep_id,
            SUM(total) dollars_spent
        FROM invoice i 
        INNER JOIN customer c ON c.customer_id = i.customer_id
        GROUP BY 1
        )
        
        
    SELECT 
        e.first_name || " " || e.last_name agent_name,    
        SUM(spc.dollars_spent) sales_amount,
        e.hire_date
    FROM employee e
    INNER JOIN sales_per_customer spc ON spc.support_rep_id = e.employee_id
    GROUP BY 1
    ORDER BY 2 DESC;  
    
'''

sales_per_agent = run_query(q2)
sales_per_agent
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 30))

ax1, ax2 = axes.flatten()
fig.subplots_adjust(hspace=0.8, wspace=3)

#Top axis
agent_names = sales_per_agent['agent_name'].tolist()
amounts = sales_per_agent['sales_amount'].tolist()
percentage_sales = np.array(amounts)/sum(amounts)
colors = ['#ff9999','#66b3ff','#99ff99']

patches, texts, autotexts = ax1.pie(percentage_sales, 
                                    colors=colors, 
                                    labels=agent_names, 
                                    autopct='%1.1f%%', 
                                    startangle=90)

centre_circle = plt.Circle((0,0), 0.85, fc='white')
axes[0].add_patch(centre_circle)

for i in range(0,3):
    texts[i].set_fontsize(45)
    autotexts[i].set_fontsize(38)
    

ax1.set_title('Sales Breakdown by Agent', fontsize=55, y=0.96)
ax1.axis('equal')


#Bottom axis
sales_per_agent.plot.bar(x='agent_name', 
                         y='sales_amount', 
                         ax=ax2, 
                         colormap=plt.cm.ocean,
                         width=0.5,
                         legend=False,
                         rot=40)


for i in ax2.spines.keys():
        ax2.spines[i].set_visible(False)
        
for p in ax2.patches:
    ax2.annotate(str(round(p.get_height())) + "$", 
                 (p.get_x() * 1.01, 
                  p.get_height() * 0.93),
                  fontsize=50, color='white', weight='bold'
                )  

ax2.tick_params(axis = 'x', labelsize = 35, top="off", left="off", right="off", bottom='off')
ax2.set_xlabel('')
y_axis = ax2.axes.get_yaxis()
y_axis.set_visible(False)
ax2.set_title('Total Sales by Agent', fontsize=55, y=1.05)

plt.subplots_adjust(hspace=1.3)
plt.tight_layout()
plt.show()
q3 = '''
    WITH 
        sales_by_country AS
        (
            SELECT 
                c.customer_id,
                CASE
                   WHEN (
                         SELECT count(*)
                         FROM customer
                         where country = c.country
                        ) = 1 THEN "Other"
                   ELSE c.country
                   END AS country,
                il.invoice_id,
                il.unit_price
            FROM invoice_line il
            INNER JOIN invoice i ON i.invoice_id = il.invoice_id
            INNER JOIN customer c ON c.customer_id = i.customer_id
            
        )
        
        
    SELECT
        country,
        number_of_customers,
        total_sales,
        avg_sale_value,
        avg_order_value
    FROM
        ( 
        SELECT
            country,
            COUNT(DISTINCT customer_id) number_of_customers,
            SUM(unit_price) total_sales,
            SUM(unit_price) / COUNT(DISTINCT customer_id) avg_sale_value,
            SUM(unit_price) / COUNT(DISTINCT invoice_id) avg_order_value,
            CASE 
                WHEN country = 'Other' THEN 1
                ELSE 0
                END AS sort
        FROM sales_by_country
        GROUP BY country
        )
    ORDER BY sort ASC, total_sales DESC;

'''

sales_by_country = run_query(q3)
run_query(q3)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22,17))
ax1, ax2, ax3, ax4 = axes.flatten()

#Top left axis
countries = sales_by_country['country'].tolist()
amounts = sales_by_country['total_sales'].tolist()
percentage_sales = np.array(amounts)/sum(amounts)
colors = cm.Set3(np.arange(10)/10.)
explode = [0]*10
explode[0] = 0.1

patches, texts, autotexts = ax1.pie(percentage_sales, 
                                    colors=colors, 
                                    labels=countries, 
                                    autopct='%1.1f%%',
                                    explode=explode,
                                    startangle=-90,
                                    shadow=True)

for i in range(0,10):
    texts[i].set_fontsize(18)
    autotexts[i].set_fontsize(0)
    
ax1.set_title('Sales Breakdown by Country', fontsize=32, y=1.1)
ax1.axis('equal') 



#Top right axis
mean_value_orders = sales_by_country['avg_order_value'].mean()
sales_by_country_copy = sales_by_country[['country', 'avg_order_value']]
differences = sales_by_country_copy['avg_order_value'] - mean_value_orders 
sales_by_country_copy['percentage_difference'] = differences/mean_value_orders
    
    
sales_by_country_copy.plot.bar(x='country', 
                               y='percentage_difference', 
                               ax=ax2, 
                               colormap=plt.cm.Reds,
                               width=0.5,
                               legend=False,
                               rot=40)

ax2.set_xlabel('')
ax2.set_ylabel('Pct Difference')
ax2.set_title('Average Order Value\nPct Difference From Mean', fontsize=30, y=1.04)
ax2.tick_params(top="off", right="off", left="off", bottom="off")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)



#Bottom left axis
ax_overlap = ax3.twinx()

sales_by_country.plot.bar(x='country', 
                         y='total_sales', 
                         ax=ax3, 
                         colormap=plt.cm.Accent,
                         width=0.5,
                         legend=False,
                         alpha=0.2,
                         rot=40)

ax_overlap.plot(sales_by_country['number_of_customers'], lw=2,marker='o')

ax3.set_title('Sales vs. Number of \nCustomers by Country', fontsize=32, y=1.05)
ax3.set_ylabel('Total Sales ($)')
ax_overlap.set_ylabel('Number of Customers')
ax3.set_xlabel('')
ax3.spines["top"].set_visible(False)
ax_overlap.spines["top"].set_visible(False)
ax3.tick_params(top="off", left="off", right="off", bottom='off')
ax_overlap.tick_params(top="off", left="off", right="off", bottom='off')

for i in range(0,7):
    ax3.get_yticklabels()[i].set_color("green")
for i in range(0,8):
    ax_overlap.get_yticklabels()[i].set_color("blue")


    
    
#Bottom right axis
sales_by_country.plot.bar(x='country', 
                         y='avg_sale_value', 
                         ax=ax4, 
                         width=0.5,
                         colormap=plt.cm.jet,
                         legend=False,
                         alpha=1,
                         rot=40)

for i in ax4.patches:
    ax4.text(i.get_x()+.15, i.get_height()-8, \
            str(round((i.get_height()))) + ' $', fontsize=20, color='white', rotation=90, weight='bold')

ax4.set_title('Sales per Customer \nAverage by Country', fontsize=32, y=1.05)
ax4.set_xlabel('')
ax4.tick_params(top="off", left="off", right="off", bottom='off')
ax4.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
y_axis = ax4.axes.get_yaxis()
y_axis.set_visible(False)

plt.subplots_adjust(hspace=1, wspace=.85)
plt.tight_layout()
plt.show()



q4 = '''
    WITH    
        album_invoice AS
                (
                SELECT 
                    il.invoice_id,
                    il.track_id,
                    t.album_id
                FROM invoice_line il
                INNER JOIN track t ON t.track_id = il.track_id
                ),
        
        
        invoice_info AS
                (
                SELECT 
                    invoice_id,
                    COUNT(DISTINCT album_id) num_albums,
                    COUNT(track_id) num_tracks,
                    CASE
                        COUNT(DISTINCT album_id)
                        WHEN 1 THEN album_id
                        ELSE NULL
                        END AS album_id
                FROM album_invoice
                GROUP BY invoice_id
                ),
            
            
        track_album AS
                (
                SELECT 
                    COUNT(track_id) num_tracks,
                    album_id
                FROM track t
                WHERE album_id IN (
                                  SELECT album_id FROM invoice_info
                                  WHERE num_albums = 1    
                                 )
                GROUP BY album_id
                ORDER BY album_id ASC
                )
          
       
       
    SELECT 
        album_purchase,
        COUNT(invoice_id) num_invoices,
        CAST(COUNT(invoice_id) as float)/(SELECT COUNT(*) FROM invoice) percent 
    FROM
      (
        SELECT
            invoice_id,
            CASE
                WHEN (ii.album_id == ta.album_id AND ii.num_tracks == ta.num_tracks) THEN 'Yes'
                ELSE 'No'
                END AS album_purchase
        FROM invoice_info ii
        LEFT JOIN track_album ta ON ii.album_id = ta.album_id
      ) 
    GROUP BY album_purchase;

'''

album_purchases = run_query(q4)
run_query(q4)
fig = plt.figure(figsize=(9,6))

cases = ['Album', 'Individual Tracks']
amounts = album_purchases['num_invoices'].tolist()
percentage_purchases = np.array(album_purchases['percent'].tolist())
explode = [0]*2
explode[0] = 0.1

colors = ['#d46231', '#872424']
patches, texts, autotexts = plt.pie(percentage_purchases, 
                                    colors=colors, 
                                    labels=cases, 
                                    explode=explode,
                                    autopct='%1.1f%%', 
                                    startangle=200
                                   )

for i in range(0,2):
    texts[i].set_fontsize(18)
    autotexts[i].set_fontsize(15)
    autotexts[i].set_color('white')
    autotexts[i].set_weight('bold')
    

plt.title('Purchases\nAlbum vs. Individual Tracks', fontsize=20, y=1.08)
plt.axis('equal')
plt.tight_layout()
plt.show()
q5 = '''
    WITH media_type_sold AS
        (
            SELECT
                il.track_id,
                sm.song,
                sm.media_type,
                SUM(il.quantity) units_sold

            FROM invoice_line il
            LEFT JOIN 
                    (
                        SELECT
                            t.track_id,
                            t.name song,
                            mt.name media_type
                        FROM media_type mt
                        INNER JOIN track t ON t.media_type_id = mt.media_type_id
                    ) sm ON il.track_id = sm.track_id

            GROUP BY il.track_id
        )
    
    
    SELECT 
        media_type,
        SUM(units_sold) tracks_sold
    FROM media_type_sold
    GROUP BY media_type
    ORDER BY tracks_sold DESC;
    
    
'''

media_type_units = run_query(q5)
run_query(q5)
ax4 = media_type_units.plot.bar(x='media_type', 
                         y='tracks_sold',  
                         width=0.5,
                         colormap=plt.cm.magma,
                         legend=False,
                         alpha=1,
                         figsize=(15,12),
                         rot=40)

for i in ax4.patches:
    ax4.text(i.get_x()+.15, i.get_height()+105, \
            str(round((i.get_height()))), fontsize=22, color='black')
    
    
ax4.set_title('Media Types - Units Sold', fontsize=26, y=1.05)
ax4.set_xlabel('')
ax4.tick_params(top="off", left="off", right="off", bottom='off')
ax4.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
y_axis = ax4.axes.get_yaxis()
y_axis.set_visible(False)

plt.subplots_adjust(hspace=1, wspace=.85)
plt.tight_layout()
plt.show()

