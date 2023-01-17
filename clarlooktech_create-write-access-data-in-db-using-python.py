import sqlite3

import pandas as pd
conn = sqlite3.connect('employee.db')
c = conn.cursor()
c.execute("""CREATE TABLE employees (

    fname text,

    lname text,

    pay real,

    id integer,

    pnumber integer )""")



conn.commit()
c.execute("INSERT INTO employees VALUES ('Hailee', 'Orniz','54000.00','002','6190352345')")



conn.commit()
c.execute("SELECT * FROM employees")

print(c.fetchall())
c.execute("INSERT INTO employees VALUES ('Savana', 'Otchika','51100.23','001','6177654432')")



conn.commit()
c.execute("INSERT INTO employees VALUES ('Henry', 'Livtcka','65200.00','003','6144332344')")



conn.commit()
#c.execute("SELECT * FROM employees WHERE lname='Orniz'")

c.execute("SELECT * FROM employees")

print(c.fetchall())
# add more employees

c.execute("INSERT INTO employees VALUES ('Hanna', 'Bucker','95000.00','004','6145472344')")

c.execute("INSERT INTO employees VALUES ('Vladimir', 'Kroz','45500.00','005','6133653231')")

c.execute("INSERT INTO employees VALUES ('Laveet', 'Senya','66060.00','006','6111897343')")

c.execute("INSERT INTO employees VALUES ('Jane', 'Doe','67889.00','007','6111133249')")

c.execute("INSERT INTO employees VALUES ('Patrick', 'Doyle','77118.00','008','6321133232')")

c.execute("INSERT INTO employees VALUES ('Sam', 'Meyers','77118.00','009','6326533211')")

c.execute("INSERT INTO employees VALUES ('Alvin', 'Mayers','56321.00','010','6326533188')")

c.execute("INSERT INTO employees VALUES ('Joana', 'Sims','46300.00','011','6399533188')")



conn.commit()
c.execute("SELECT * FROM employees")

print(c.fetchall())
# remove duplicate id's

#c.execute("DELETE FROM employees WHERE id='1'")

#print(c.fetchall())

c.execute("SELECT * FROM employees")

print(c.fetchall())
conn2 = sqlite3.connect('orders.db')
c2 = conn2.cursor()
c2.execute("""CREATE TABLE orders (

    id integer,

    product_name text,

    amount real,

    customer_id integer

    )""")

conn.commit()
c2.execute("INSERT INTO orders VALUES ('1', 'Yoga Mat','23.10','12')")

c2.execute("INSERT INTO orders VALUES ('6', 'Fossil Watch','303.10','10')")

c2.execute("INSERT INTO orders VALUES ('5', 'Hann Dog Toy','10.00','10')")

c2.execute("INSERT INTO orders VALUES ('9', 'Kitchen Knife','15.50','2')")

c2.execute("INSERT INTO orders VALUES ('3', 'Laptop Cover','5.30','3')")

c2.execute("INSERT INTO orders VALUES ('2', 'Mug','6.10','5')")

conn.commit()
c2.execute("SELECT * FROM orders")

print(c2.fetchall())
employeesdb = pd.read_sql_query("select * from employees limit 100;", conn)

print(employeesdb.head())
ordersdb = pd.read_sql_query("select * from orders limit 100;", conn2)

print(ordersdb.head())
query = '''

    SELECT employees.fname, employees.lname, employees.pnumber, orders.product_name, orders.amount

    FROM employees INNER JOIN orders

    ON employees.id=orders.customer_id

    

    '''



#then use read_sql_squery method



## df = pd.read_sql_query(query, engine)

##df