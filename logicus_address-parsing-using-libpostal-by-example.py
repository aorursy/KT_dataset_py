# Installing dependencies

!conda install -c conda-forge postal -y
# Address parser using Libpostal



from postal.parser import parse_address

from postal.expand import expand_address
import json
def convert_json(address):

    new_address = {k: v for (v, k) in address}

    json_address = json.dumps(new_address, sort_keys=True,ensure_ascii = False, indent=1)

    return json_address
def address_parser(address):

    

    # I use first position as default even though sometimes expand address returns more than one

    # Expand address tries to expand some 

    expanded_address = expand_address( address )[0]

    parsed_address = parse_address( expanded_address )

    json_address = convert_json(  parsed_address )

    

    return json_address

    

#     return convert_json(

#         parse_address( 

#             expand_address(address)[0] 

#             )

#         )
print(address_parser("10600 N Tantau Ave")) # Apple Address