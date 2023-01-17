from lxml import etree

url_iatistandard_org = 'http://datastore.iatistandard.org/api/1/access/activity.xml'

# https://lxml.de/parsing.html

root = etree.parse(url_iatistandard_org)
root
etree.tostring(root)
# print(etree.tostring(root, pretty_print=True))

# https://www.journaldev.com/18043/python-lxml

print(etree.tostring(root, pretty_print=True).decode("utf-8"))
result = root.getroot()
result
# https://python101.pythonlibrary.org/chapter31_lxml.html

for appt in result.getchildren():

        for elem in appt.getchildren():

            #print(elem)

            print('elem.tag : ' + elem.tag)

            print('elem[0].tag : ' + elem[0].tag)

            print('elem[0].text : ' + elem[0].text)
# https://python101.pythonlibrary.org/chapter31_lxml.html

for appt in result.getchildren():

        for elem in appt.getchildren():

            for in_elem in elem.getchildren():

                print(in_elem.tag)

                if in_elem.text:

                    if isinstance(in_elem.text, str):

                        print('in_elem.text : '+ in_elem.text)
result.getchildren()[0].getnext().tag
# https://lxml.de/api.html

[ el.tag for el in root.iter() ]
len([ el.tag for el in root.iter() ])
[ el.text for el in root.iter('narrative') ]
len([ el.text for el in root.iter('narrative') ])
[ el.text for el in root.iter('participating-org') ]
len([ el.text for el in root.iter('participating-org') ])
[ el.text for el in root.iter('title') ]
[ el.text for el in root.iter('description') ]
[ el.text for el in root.iter('title', 'participating-org') ]