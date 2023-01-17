def get_rel_wahr(abhVarWerte):

    anzahlWerte = len(abhVarWerte)

    relWahr = {}

    # relative Haeufigkeiten bestimmen = relative Wahrscheinlichkeiten

    for auspraegung in abhVarWerte:

        if auspraegung in relWahr:

            relWahr[auspraegung] = relWahr[auspraegung] + 1/anzahlWerte

        else:

            relWahr[auspraegung] = 1/anzahlWerte

    return relWahr



def get_werte(dataFrame, bedingungen):

    dataFrameBed = dataFrame

    for variable, bed in bedingungen.items(): 

        dataFrameBed = dataFrameBed[dataFrameBed.loc[:, variable] == bed]           

    return dataFrameBed    



def get_auspr(dataFrame, variable):

    return dataFrame.loc[:, variable].drop_duplicates()



def get_gini_impurity(rel_wahr):

    # Gini Impurity bestimmen

    sq_prob_sum = 0

    for wahrschl in rel_wahr:

        sq_prob_sum += wahrschl*wahrschl

    gini_impurity = 1 - sq_prob_sum

    return gini_impurity
import pandas as pd

data = {'Height': ['Small', 'Tall', 'Tall', 'Tall', 'Small', 'Tall', 'Tall', 'Small'], 'Hair': ['Blonde', 'Dark', 'Blonde', 'Dark', 'Dark', 'Red', 'Blonde', 'Blonde'], 'Eyes': ['Brown', 'Brown', 'Blue', 'Blue', 'Blue', 'Blue', 'Brown', 'Blue'], 'Attractive': ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']}

#data = {'Deadline': ['Dringend', 'Dringend', 'Bald', 'Nein', 'Nein', 'Nein', 'Bald', 'Bald', 'Bald', 'Dringend'], 'Party': ['Ja', 'Nein', 'Ja', 'Ja', 'Nein', 'Ja', 'Nein', 'Nein', 'Ja', 'Nein'], 'Faul': ['Ja', 'Ja', 'Ja', 'Nein', 'Ja', 'Nein', 'Nein', 'Ja', 'Ja', 'Nein'], 'Aktivität': ['Party', 'Lernen', 'Party', 'Party', 'Kneipe', 'Party', 'Lernen', 'TV', 'Party', 'Lernen']}

#data = {'Season': ['Summer', 'Winter', 'Winter', 'Summer', 'Spring', 'Spring'], 'Temperature': ['Warm', 'Cold', 'Warm', 'Cold', 'Warm', 'Cold'], 'Eat Ice-Cream': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No']}

#data = {'Computer': ['Running', 'Standby', 'Off', 'Standby', 'Running', 'Off'], 'Projector': ['On', 'On', 'Off', 'Off', 'Off', 'On'], 'Temperature outside': ['Warm', 'Hot', 'Warm', 'Hot', 'Cold', 'Hot'], 'Temperature inside': ['Warm', 'Hot', 'Warm', 'Warm', 'Warm', 'Hot']}

#data = {'Time': ['Morning', 'Evening', 'Night', 'Morning', 'Evening', 'Evening'], 'Last Meal': ['Dinner', 'Lunch', 'Dinner', 'Lunch', 'Breakfast', 'Lunch'], 'Hungry': ['Hungry', 'Not Hungry', 'Not Hungry', 'Very Hungry', 'Very Hungry', 'Hungry']}

dataFrame_input = pd.DataFrame(data=data)

print(dataFrame_input)



header = list(dataFrame_input)

abh_var = header[len(header)-1]

header.remove(abh_var)

unabh_var = header
from scipy.stats import entropy



def start_build_decision_tree(dataFrame, unabh_var_templ, method):

    if method == 'information_gain':

        method = 0

    elif method == 'gini_impurity':

        method = 1

    build_decision_tree(dataFrame, unabh_var_templ, method, -1)



def build_decision_tree(dataFrame, unabh_var_templ, method, last_entropy):

    # Die Eingabedaten sollen nicht verändert werden

    ungenutzte_unabh_var = unabh_var_templ.copy()

    

    # Dafür die relativen Wahrscheinlichkeiten für die Ausprägungen von Attractive bestimmen

    abh_var_werte = dataFrame.get(abh_var).values

    rel_wahr = get_rel_wahr(abh_var_werte)

    print("Relative Wahrscheinlichkeiten: %s" % rel_wahr)

    

    if last_entropy == 0 or len(rel_wahr) == 1:

        rel_wahr = list(rel_wahr)

        print("==> Wähle '%s'" % rel_wahr[0])

        return

    

    rel_wahr = list(rel_wahr.values())       



    if last_entropy == -1:

        if method == 0:

            # Entropy bestimmen

            entropy_value = entropy(rel_wahr, qk=None, base=2)

            #print("=> Entropy: %f" % entropy_value)

        

        elif method == 1:

            # Gini Impurity bestimmen

            entropy_value = get_gini_impurity(rel_wahr)

            #print("=> Gini Impurity: %f" % entropy_value)

    else:

        entropy_value = last_entropy

        

    if len(ungenutzte_unabh_var) == 1:

        chosen_unabh_var = ungenutzte_unabh_var[0]

        print("==> Wähle '%s'" % chosen_unabh_var)

        ungenutzte_unabh_var.remove(chosen_unabh_var)

        jump_to_next_stage(dataFrame, chosen_unabh_var, ungenutzte_unabh_var, method, -1, entropy_value)   

        return

    

    print("=> Entropy: %s" % entropy_value)

        

    gains = {}

    new_entropies = {}

    for variable in ungenutzte_unabh_var:

        #print("\n%s:" % variable)

        var_feat = dataFrame.get(variable).values



        rel_haeuf = get_rel_wahr(var_feat)

        #print("Relative Häufigkeiten: %s" % rel_haeuf)



        new_entropies[variable] = {}

        sum = 0

        for auspr,haeuf in rel_haeuf.items():

            dict = {variable: auspr}

            werte = get_werte(dataFrame, dict)

            rel_wahr = get_rel_wahr(werte.get(abh_var).values)

            #print("Relative Wahrscheinlichkeiten (%s): %s" % (auspr, rel_wahr))

            rel_wahr = list(rel_wahr.values())

            

            if method == 0:

                new_entropy = entropy(rel_wahr, qk=None, base=2)                

                

            elif method == 1:

                new_entropy = get_gini_impurity(rel_wahr)

       

            new_entropies[variable][auspr] = new_entropy

            sum += haeuf * new_entropy



        # Gain bestimmen (Information Gain oder Gini Gain)

        gain = entropy_value - sum

        #print("Gain: %s" % gain)      

        

        gains[variable] = gain



    print("\nGain Werte:\n", gains)

         

    key_max = max(gains.keys(), key=(lambda k: gains[k]))   

    

    print("==> Wähle '%s'" % key_max)

    ungenutzte_unabh_var.remove(key_max)

    jump_to_next_stage(dataFrame, key_max, ungenutzte_unabh_var, method, new_entropies, -1)

    

def jump_to_next_stage(dataFrame, key_max, ungenutzte_unabh_var, method, new_entropies, entropy_val):

    for auspr in get_auspr(dataFrame, key_max):

        print("\n-> Zweig: %s" % auspr)

        dict = {key_max: auspr}

        dF_naechste_ebene = get_werte(dataFrame, dict)

        #print(dF_naechste_ebene)

        #print(new_entropies)

        build_decision_tree(dF_naechste_ebene, ungenutzte_unabh_var, method, entropy_val if entropy_val != -1 else new_entropies[key_max][auspr]) 
start_build_decision_tree(dataFrame_input, unabh_var, 'information_gain')
start_build_decision_tree(dataFrame_input, unabh_var, 'gini_impurity')