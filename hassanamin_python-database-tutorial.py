# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sqlite3

con = sqlite3.connect('../input/census.sqlite')

from sqlalchemy import create_engine



engine = create_engine('sqlite:///../input/census.sqlite')
print(engine.table_names())
from sqlalchemy import MetaData, Table

metadata = MetaData()

census = Table('census', metadata, autoload=True, autoload_with=engine)

print(repr(census))
stmt = 'Select * from census'

result_proxy = engine.execute(stmt)

results = result_proxy.fetchall()

results
print('First Row : ', results[0])

print('Keys : ',results[0].keys())

print('Keys : ',results[0].state)
from sqlalchemy.sql import select

stmt = select([census])

stmt = stmt.where(census.columns.state == 'California')

results = engine.execute(stmt).fetchall()



for result in results:

    print(result.state, result.age)
stmt = select([census])

stmt = stmt.where(census.columns.state.startswith('New'))

results = engine.execute(stmt).fetchall()



for result in results:

    print(result.state, result.age)
stmt = select([census])

stmt = stmt.where(census.columns.state.startswith('New'))

#results = engine.execute(stmt).fetchall()



for result in engine.execute(stmt).fetchall():

    print(result.state, result.age)
from sqlalchemy import or_

stmt = select([census])



stmt = stmt.where(or_(census.columns.state == 'California',census.columns.state == 'New York'))



for result in engine.execute(stmt):

    print(result.state, result.sex)
stmt = select([census.columns.state])

stmt = stmt.order_by(census.columns.state)

results = engine.execute(stmt).fetchall()

print(results[:10])
stmt = select([census.columns.state, census.columns.sex])

stmt = stmt.order_by(census.columns.state, census.columns.sex)

results = engine.execute(stmt).first()

print(results)
from sqlalchemy import func

stmt = select([func.sum(census.columns.pop2008)])

results = engine.execute(stmt).scalar()

print(results)
stmt = select([census.columns.sex, func.sum(census.columns.pop2008)])

stmt = stmt.group_by(census.columns.sex)

results = engine.execute(stmt).fetchall()

print(results)
stmt = select([census.columns.sex, census.columns.age, func.sum(census.columns.pop2008)])

stmt = stmt.group_by(census.columns.sex, census.columns.age)

results = engine.execute(stmt).fetchall()

print(results)
print(results[0].keys())

stmt = select([census.columns.sex, func.sum(census.columns.pop2008).label('pop2008_sum')])

stmt = stmt.group_by(census.columns.sex)

results = engine.execute(stmt).fetchall()

print(results[0].keys())
import pandas as pd

df = pd.DataFrame(results)

df.columns = results[0].keys()

print(df)
import matplotlib.pyplot as plt

df.plot.barh()

plt.show()
from sqlalchemy import desc

stmt = select([census.columns.age,(census.columns.pop2008 - census.columns.pop2000).label('pop_change')])

stmt = stmt.group_by(census.columns.age)

stmt = stmt.order_by(desc('pop_change'))

stmt = stmt.limit(5)

results = engine.execute(stmt).fetchall()

print(results)
from sqlalchemy import case

stmt = select([func.sum(case([(census.columns.state == 'New York',census.columns.pop2008)], else_=0))])

results = engine.execute(stmt).fetchall()

print(results)
from sqlalchemy import case, cast, Float

stmt = select([(func.sum(case([(census.columns.state == 'New York',census.columns.pop2008)], else_=0)) /cast(func.sum(census.columns.pop2008),

                                                                        Float) * 100).label('ny_percent')])

results = engine.execute(stmt).fetchall()

print(results)
## Reflection for state_fact

state_fact = Table('state_fact', metadata, autoload=True, autoload_with=engine)

print(repr(state_fact))
stmt = select([census.columns.pop2008, state_fact.columns.abbreviation])

results = engine.execute(stmt).fetchall()

print(results)
stmt = select([func.sum(census.columns.pop2000)])

stmt = stmt.select_from(census.join(state_fact, census.columns.state == state_fact.columns.name))

stmt = stmt.where(state_fact.columns.census_division_name == 'East South Central')

result = engine.execute(stmt).scalar()

print(result)