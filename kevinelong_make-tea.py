def fill_kettle(kettle):
    kettle.append("water")
    
def get_hotwater():
    k = []
    fill_kettle(k)
    return "hot water"

def make_tea():
    saucer = []
    water = get_hotwater();
    return saucer
# STEPS

# Heat the water.

#     Get the kettle/pot.
#     Go to sink.
#     turn on the faucet.
#     place kettle under the water.
#     when it full then remove kettle from faucet.
#     and turn off the faucet

# Put kettle on stove.
#     turn on corresponding burner.
#     wait until it whistles/boils.
#     turn off burner

# Get a teacup from the cupboard
# Put tea bag of select type into the cup.
# put in extras into cup.
# Pour water from kettle into cup until 80% full.

# stir with bag or spoon

# wait until brewed to desired strengh (minutes: light=3 or dark=5)

# place tea cup on saucer
# gently and carfully bring to guest.


def get_it_done():
    return "Done!"

print(get_it_done())


kettle = []
kettle.append("drop of water")
kettle.append("drop of water")
kettle.append("drop of water")
kettle.append("drop of water")
print(len(kettle))
print(kettle)

# kind = "earl grey"
kind = None
print(kind)

if kind is not None:
    print(f"Making {kind}")
else:
    print("WHat kind would you like?")
