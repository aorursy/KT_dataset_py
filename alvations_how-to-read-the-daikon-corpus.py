import json



# Load the Daikon Corpus into memory.

with open('../input/daikon-v1.json') as fin:

    json_data = json.load(fin)
# Iterate through the JSON object. 

for article in json_data: 

    # Format the JSON into humanly pretty strings.

    pretty_json = json.dumps(article, indent=4, sort_keys=True)

    print(pretty_json)

    break