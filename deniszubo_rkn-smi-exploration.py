import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xml.etree.ElementTree as ET
import io
input_path = '../input/data-20180514T0000-structure-20160909T0000.xml'

parsedXML = ET.parse(input_path)
ns = {'rkn': 'http://rsoc.ru/opendata/7705846236-ResolutionSMI'}
nodes = parsedXML.getroot()
names = []
for node in nodes:
    for record in node.findall('rkn:name', ns):
        names.append(record.text)

df = pd.DataFrame({'name': names})
display(df.shape)
display(df.head())
