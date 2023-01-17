!pip install rdflib

!pip install pymarc
import datetime

import io

import rdflib

import requests

import pymarc

import sys

from lxml import etree

print(datetime.datetime.utcnow())



BF = rdflib.Namespace("http://id.loc.gov/ontologies/bibframe/")

BFLC = rdflib.Namespace("http://id.loc.gov/ontologies/bflc/")

SINOPIA = rdflib.Namespace("http://sinopia.io/vocabulary/")

XML_NS = {"bf": "http://id.loc.gov/ontologies/bibframe/"}

# Function binds three of the specific RDF namespaces used by Sinopia.

def init_ns(graph):

    graph.namespace_manager.bind("bf", BF)

    graph.namespace_manager.bind("bflc", BFLC)

    graph.namespace_manager.bind("sinopia", SINOPIA)

    

# Function takes a list Sinopia URIs and returns a RDF Graph

def triples2record(uris):

    record_graph = rdflib.Graph()

    init_ns(record_graph)

    for uri in uris:

        get_request = requests.get(uri)

        raw_utf8 = get_request.text.encode(get_request.encoding).decode()

        record_graph.parse(data=raw_utf8, format='turtle')

    return record_graph



# Function takes a graph, serializes to XML, and the returns the transformed MARC XML

def graph2marcXML(graph):

    rdf_xml = etree.XML(graph.serialize(format='pretty-xml'))

    try:

        return bf2marc_transform(rdf_xml)

    except:

        print(f"Error transforming graph to {sys.exc_info()[0]}")

    

# Function takes a list of Sinopia URLs, creates graphs for each, serializes to XML before, combining

# back into a single XML document for transformation 

def graph2RDFXML(uris):

    for uri in uris:

        graph = triples2record(uri)

        rdf_xml = etree.XML(graph.serialize(format='pretty-xml'))

        yield rdf_xml



# Function takes a MARC XML document and returns the MARC 21 equalivant

def marcXMLto21(marcXML):

    reader = io.StringIO(etree.tostring(marcXML).decode())

    marc_records = pymarc.parse_xml_to_array(reader)

    # Should only be 1 record in the list

    return marc_records[0]

    

# Function takes a Sinopia URI that has a relative URI, replaces relative with absolute, and returns 

# new graph

def update_abs_url(uri):

    get_request = requests.get(uri)

    raw_utf8 = get_request.text.encode(get_request.encoding).decode()

    raw_utf8 = raw_utf8.replace('<>', f"<{uri}>")

    graph = rdflib.Graph()

    init_ns(graph)

    graph.parse(data=raw_utf8, format='turtle')

    return graph



# Function takes a graph with a URI bf:Work with a nested bf:Instance serializes the output to XML and then modifies 

# XML with the expected structure for the bibframe2marc transformation

def nestedInstance(graph):



    rdf_xml = etree.XML(graph.serialize(format='pretty-xml'))

    hasInstance = rdf_xml.find("bf:Work/bf:hasInstance", XML_NS)

    bfInstance = hasInstance.find("bf:Instance", XML_NS)

    # Delete the instance from the hasInstance element

    hasInstance.remove(bfInstance)

    # Creates relationship betweemn bf:hasInstance and the instance

    node_id = bfInstance.attrib["{http://www.w3.org/1999/02/22-rdf-syntax-ns#}nodeID"]

    hasInstance.attrib["{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"] = node_id

    # Adds rdf:about to bfInstance

    bfInstance.attrib["{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"] = node_id

    # Adds the instance back as a top-level element

    rdf_xml.append(bfInstance)

    return rdf_xml



# Function takea a graph with an bf:Instance with one or more nested bf:Works, serializes to XML and then modifies

# the XML to the expected structure for the bibframe2marc xslt

def nestedWork(graph):

    rdf_xml = etree.XML(graph.serialize(format='pretty-xml'))

    instanceOfs = rdf_xml.findall("bf:instanceOf", XML_NS)

    for elem in instanceOfs:

        bfWork = elem.find("bf:Work", XML_NS)

        work_node_id = bfWork.attrib["{http://www.w3.org/1999/02/22-rdf-syntax-ns#}nodeID"]

        elem.attrib["{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"] = work_node_id

        elem.remove(bfWork)

        rdf_xml.append(bfWOrk)

    return rdf_xml



# Opens the XSLT as a file and creates an lxml XSLT object.

with open("../input/bibframe2marcxsl/bibframe2marc.xsl", "rb") as fo:

    xml = etree.XML(fo.read())

    bf2marc_transform = etree.XSLT(xml)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/94aeb119-7ded-4288-975a-4e83c6d3121a'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/dc417083-8a87-4235-9352-eae9efdbe705'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/cc719140-8657-4404-bdbb-c5f93ce9a011'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/b6a11033-8812-496f-a199-c10f0e9078d6'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/cc719140-8657-4404-bdbb-c5f93ce9a011'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/8c6b6660-ea4a-4b6b-89b1-c5a6476dcc7a'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/31a97c7f-2f65-4d4d-9d43-88f54519dc92'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/48921a54-f527-4cb6-97ca-6668e725a80f'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/d92eaa9a-e35d-4e7a-99d3-f7b5e951285a'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/dd1356ad-cec0-4626-814f-ba7811ca2440'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/8571c8a6-4374-405c-a979-0d7241816786'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/4d0606bd-76be-441c-af6a-09baa42280ed'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/8571c8a6-4374-405c-a979-0d7241816786'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/8aecd451-cefa-479a-8d58-cc87f17af719'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/33d1b00a-7678-4901-af07-32537ebfa84d'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/d6b74251-1532-477d-942c-92802c7cb2ba'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/d3df3bdc-bcdd-47b5-a9d3-808039386848'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/a7a177fa-f52c-4180-b673-7b92ca579c5b'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/8c043c0f-2fe8-44c9-8524-c41e7f669a7f'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/97f8a2fb-0d0a-47da-ac48-bb6d27b2c791'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/a414dc5e-2b10-412a-8263-094aa0077bad'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/1d2954d5-e959-4899-84f9-cef2e90e8fa8'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)
work_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/4319ad83-c7bb-49d7-90bc-1e47fc8a234e'])

instance_graph = triples2record(['https://trellis.stage.sinopia.io/repository/penn/9826b46d-7274-4b20-9dc9-535a2c1e0ea5'])

record_xml = etree.XML(instance_graph.serialize(format='pretty-xml'))
work_xml = etree.XML(work_graph.serialize(format='pretty-xml'))

work_element = work_xml.find("bf:Work", XML_NS)

for child in work_xml.iterchildren():

    record_xml.append(child)

print(etree.tostring(record_xml).decode())
marc_xml = bf2marc_transform(record_xml)

marc21 = marcXMLto21(marc_xml)
print(marc21)