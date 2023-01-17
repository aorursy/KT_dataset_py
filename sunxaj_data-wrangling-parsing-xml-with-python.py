# Code to import libraries
import re
import json
# Read the file
file = open("../input/australian-sport-thesaurus-student.xml", encoding = 'utf-8')
# Explore the file content
for text in file:
    print(text)
file = open("australian-sport-thesaurus-student.xml", encoding = 'utf-8')

terms = []
rootTerm = {}
relatedTerm = []
childTerm = {}
isRootTerm = False

rootTermMatch = re.compile(r"^\s*<Terms>$")
endOfTermMatch = re.compile(r"^\s*</Term>\s*")
titleMatch = re.compile(r"^\s*<Title>.*")
descriptionMatch =re.compile(r"\s*<Description>.*")
relatedTermsMatch = re.compile(r"\s*<RelatedTerms>.*")
endOfRelatedTermMatch = re.compile(r"^\s*</RelatedTerms>\s*")
relationshipMatch = re.compile(r"\s*<Relationship>.*")



for line in file:
    if rootTermMatch.match(line) != None:       # Hits the rootTerm tag: <Term>
        isRootTerm = True

    if isRootTerm and titleMatch.match(line) != None:
        rootTerm['Title'] = re.sub(r'(<Title>)|(</Title>)', '', line).strip()    # Remove the tags and white spaces
        
    if descriptionMatch.match(line) != None:
        rootTerm['Description'] = re.sub(r'(<Description>)|(</Description>)', '', line).strip()
        
    if relatedTermsMatch.match(line) !=None:     # Hits the childTerm tag after <RelatedTerms> tag
        isRootTerm = False
        rootTerm['RelatedTerms'] = []
    
    if not isRootTerm and titleMatch.match(line):
        childTerm['Title'] = re.sub(r'(<Title>)|(</Title>)', '', line).strip()
    
    if not isRootTerm and relationshipMatch.match(line):
        childTerm['Relationship'] = re.sub(r'(<Relationship>)|(</Relationship>)', '', line).strip()        
    
    if not isRootTerm and endOfTermMatch.match(line):
        childTerm = dict(sorted(childTerm.items(), key=lambda d:d[0]))  # Sort the dictionay by key
        rootTerm['RelatedTerms'].append(childTerm)                      # Add RelatedTerms list to the RootTerm dictionary
        childTerm = {}
        
        
    if endOfRelatedTermMatch.match(line):
        isRootTerm = True
    
    if isRootTerm and endOfTermMatch.match(line):
        rootTerm = dict(sorted(rootTerm.items(), key=lambda d:d[0]))  # Sort the dictionay by key 
        terms.append(rootTerm)                                        # Add RelatedTerms list to the Main dictionary
        rootTerm = {}
        
# Put everything into the output dcitionay and dump to a json file
dct = {"thesaurus" : terms}

with open('sport.dat', 'w') as fp:
    json.dump(dct, fp)
# Total Terms:
len(terms)