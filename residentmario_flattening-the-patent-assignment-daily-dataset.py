from lxml import etree

import pandas as pd

import numpy as np
patents = etree.parse("../input/ad20161018.xml")
root = patents.getroot()
assignments = list(list(root)[2])
assignments[:5]
len(assignments)
print(etree.tostring(assignments[0], pretty_print=True, encoding=str))
def serialize(assn):

    srs = pd.Series()

    # Metadata

    srs['last-update-date'] = assn.find("assignment-record").find("last-update-date").find("date").text

    srs['recorded-date'] = assn.find("assignment-record").find("recorded-date").find("date").text

    srs['patent-assignors'] = "|".join([assn.find("name").text for assn in assn.find("patent-assignors")])

    srs['patent-assignees'] = "|".join([assn.find("name").text for assn in assn.find("patent-assignees")])

    # WIP---below.

    try:

        srs['patent-numbers'] = "|".join(

            ["|".join([d.find("doc-number").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )

    except AttributeError:

        pass

    try:

        srs['patent-kinds'] = "|".join(

            ["|".join([d.find("kind").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )

    except AttributeError:

        pass

    try:

        srs['patent-dates'] = "|".join(

            ["|".join([d.find("date").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )    

    except AttributeError:

        pass

    try:

        srs['patent-countries'] = "|".join(

            ["|".join([d.find("country").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )    

    except AttributeError:

        pass



    try:

        srs['title'] = "|".join(

            [p.find("invention-title").text for p in assn.find("patent-properties").findall("patent-property")]

        )        

    except AttributeError:

        pass

    return srs
flattened = pd.concat([serialize(assn) for assn in assignments], axis=1).T
flattened.head()