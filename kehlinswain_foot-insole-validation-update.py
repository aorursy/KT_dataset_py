from IPython.display import HTML
HTML('<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQc-sftR4kiPCcB2-an-EsBw5y3sPbIfJNJB-nHszm7dF2obEpsS78epUBw4NVza7qdEF17QF29wJnc/embed?start=true&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>')
class InsoleRange(object):
    def __init__(self, name="Unknown name", upperLength=0, lowerLength=0):
        self.upperLength = upperLength
        self.lowerLength = lowerLength
        self.name = name
#setinsole ranges in between 5 and 10         
myInsole = InsoleRange(upperLength=10, lowerLength=5)
#Create object to contain input parameters of client (Brand) foot insole parameters
#combine dictionary parameters into one big json chunk (clear code)
insole = {"fw":int, "rfw":int, "fml":int, "fthml":int, "ah":int, "hhfm":int, "fl":int, 
                "isInsoleValid":True}
#Foot insole difference holder 
foot_insole_difference = {"fw":int, "rfw":int, "fml":int, "fthml":int, "ah":int, "hhfm":int, "fl":int}
#foot parameter key dictionary 
foot_parameter_key_dictionary = {"fw":"Foot Meterasel", "rfw":"Rear Foot Width", "fml":"First Metatarsel Length",
                                 "fthml":"Fifth Metarsel Length", "ah":"Arch Height",
                                 "hhfm":"Heel to head of First Metatarsel Phalangeal Joint",  "fl":"Foot Length"}

#Add data input gui here 
from ipywidgets import widgets
from IPython.display import display

#Brute Force Methodology Iterate dimensions in the futre
footData = widgets.Text(description="Foot Metatarsel")
footData2 = widgets.Text(description="Rear Foot Width")
footData3 = widgets.Text(description="First Metatarsel Length")
footData4 = widgets.Text(description="Fifth Metarsel Length")
footData5 = widgets.Text(description="Arch Height")
footData6 = widgets.Text(description="Heel to Head First Metatarsel Philangeal Joint")
footData7 = widgets.Text(description="Foot Length")
display(footData, footData2, footData3, footData4, footData5, footData6, footData7)
    
def handle_submit(sender):
    insole["fw"] = int(footData.value)
    
def handle_submit2(sender):
    insole["rfw"] = int(footData2.value)

def handle_submit3(sender):
    insole["fml"] = int(footData3.value)

def handle_submit4(sender):
    insole["fthml"] = int(footData4.value)
    
def handle_submit5(sender):
    insole["ah"] = int(footData5.value)

def handle_submit6(sender):
    insole["hhfm"] = int(footData6.value)
    
def handle_submit7(sender):
    insole["fl"] = int(footData7.value)
#complete.on_submit(handle_submit)
#print(insole)
footData.on_submit(handle_submit)
footData2.on_submit(handle_submit2)
footData3.on_submit(handle_submit3)
footData4.on_submit(handle_submit4)
footData5.on_submit(handle_submit5)
footData6.on_submit(handle_submit6)
footData7.on_submit(handle_submit7)
#Todo create upper and lower lengths to every foot variable 
lower_foot_ranges= {"fw":9, "rfw":7, "fml":6, "fthml":7, "ah":9, "hhfm":12, "fl":3}
upper_foot_ranges = {"fw":10, "rfw":12, "fml":15, "fthml":19, "ah":8, "hhfm":13, "fl":18}

#print(insole) Debug state
#computation for finding the parameters that need to be updated based on foot dimensions
def inSoleValidator(name,lowerLength, upperLength):
    #checks if number is outside of then it is in invalid
    if insole[name] < lowerLength or insole[name] > upperLength:
        if insole[name] < lowerLength:
            foot_insole_difference[name] = upperLength - insole[name] 
            print(foot_parameter_key_dictionary[name],"Add", foot_insole_difference[name])
        if insole[name] > lowerLength: 
            foot_insole_difference[name] = upperLength - insole[name]
            print(foot_parameter_key_dictionary[name],"Reduce", foot_insole_difference[name])
    #use case if the value is true or not
    else: print(foot_parameter_key_dictionary[name], "measurment is valid")

#iterates through each of the inputs, and runs it through the validator functions
for i in foot_insole_difference:
    lowerLength = lower_foot_ranges[i]
    upperLength = upper_foot_ranges[i]
    inSoleValidator(i,lowerLength, upperLength)
        
    
HTML('<div style="width:100%;height:0;padding-bottom:55%;position:relative;"><iframe src="https://giphy.com/embed/26h0q99fxHrZaMWm4" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/26h0q99fxHrZaMWm4">via GIPHY</a></p>')
a = 3    # number is outside of then it is not valid
a < 5 or a > 10
#tab_contents = ['P0', 'P1', 'P2', 'P3', 'P4']
children = []
for key,value in foot_parameter_key_dictionary.items():
    name = value
    children.append(widgets.Text(description=name))
#children = [widgets.Text(description=name) for name in foot_parameter_key_dictionary]
tab = widgets.Tab()
tab.children = children
for i in range(len(children)):
    tab.set_title(i, str(i))
display(tab)

def handle_submit(sender):
    print(tab.children)

tab.on_submit(handle_submit)

print(tab.children)


complete = widgets.Button(
    description='Click me',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check'
)