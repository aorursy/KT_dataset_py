data = {}

data["fms"] = "6354065120"
print( data["fms"] )
print( data["fms"][0] )
print( data["fms"][1:4] )
print( data["fms"][-7:-1] )
print( data["fms"][0:8] )
data["fmsZuKurz"] = "1234"

print( data["fmsZuKurz"][0:8] )
print("FMS Kennung: \t{0}".format(data["fms"][0:8]))

print("Status: \t{0}".format(data["fms"][8]))

print("Richtung: \t{0}".format(data["fms"][9]))