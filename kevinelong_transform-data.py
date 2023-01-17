#  list of dicts
stocks = [
    {
        "name": "TMUS",
        "closing": "200.00",
        "date": "2020-12-30",
    },
    {
        "name": "TMUS",
        "closing": "220.00",
        "date": "2020-12-31",
    },
    {
        "name": "MSFT",
        "closing": "100.00",
        "date": "2020-12-30",
    },
    {
        "name": "MSFT",
        "closing": "110.00",
        "date": "2020-12-31",
    }
]
# TODO CONVERT THE ABOVE TO THE BELOW
# what is the simplest baby step?
# loop through stocks. print?
output_dict = {}

for item_dict in stocks: #  go through all the stocks pull out one dict at a time
    name = item_dict["name"] 
    
    if name not in output_dict: # is this the first time we have seen this ticker symbol?
        output_dict[name] = [] # define ticker symbol as key and init to empty list
        
    #always safe to append to the list as we are guaranteed its initialized
    output_dict[name].append(item_dict)
    
    del item_dict["name"]
    
    print(name)
print(output_dict)

#  dict of lists of dicts
expected_output = {
    "TMUS": [
        {
            "closing": "200.00",
            "date": "2020-12-30",
        },
        {
            "closing": "220.00",
            "date": "2020-12-31",
        }
    ],

    "MSFT": [
        {
            "closing": "100.00",
            "date": "2020-12-30",
        },
        {
            "closing": "110.00",
            "date": "2020-12-31",
        }
    ]
}

#  list of dicts
stocks = [
    {
        "name": "TMUS",
        "closing": "200.00",
        "date": "2020-12-30",
    },
    {
        "name": "TMUS",
        "closing": "220.00",
        "date": "2020-12-31",
    },
    {
        "name": "MSFT",
        "closing": "100.00",
        "date": "2020-12-30",
    },
    {
        "name": "MSFT",
        "closing": "110.00",
        "date": "2020-12-31",
    }
]

output_dict = {}  # Create empty dictionary to hold out output
for s in stocks:  # Loop through all the stocks in the list
    name = s["name"]
    if name not in output_dict:  # is key not in output dict? better create the key and assign empty list
        output_dict[name] = []  # Initialize new key in dict with empty list
    output_dict[name].append(s)
    del s["name"]
print(output_dict)
assert output_dict == expected_output
