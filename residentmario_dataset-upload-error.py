import numpy as np

import pandas as pd
from lxml import etree
patents = etree.parse("../input/ipg161011.xml")
with open("../input/ipg161011.xml") as f:

    patents_text = f.read()
patents_text[:100]
d = []

s = ""

f = open("../input/ipg161011.xml")

for l in f:

    if l == "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n":

        if len(s)>0:

            d.append(s)

        s = ""

    s += l

d.append(s)



s = ""



for xm in d:

    root = etree.fromstring(xm)

    for e in root.iter(tag="invention-title"):

        s += str(e.text) + " " #e.tag, e.text, e.attrib