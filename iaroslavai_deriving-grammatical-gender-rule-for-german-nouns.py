baseline_rule = {
    'f': ['e', 'ie', 'heit', 'ei', 'in', 'ik', 'keit', 'schaft', 'ung', 'tät', 'ur', 'tion'],
    'm': ['er', 'el', 'ling', 'ich', 'ig', 'ner', 'ismus', 'or', 'us', 'eich', 'ant'],
    'n': ['chen', 'o', 'lein', 'en', 'il', 'ma', 'tel', 'ment', 'nis', 'tum', 'um']
}
import json

# read the noun list
nouns = json.load(open('../input/de_frequent_nouns.json', 'r'))

# simplify the list to the list of tuples
nouns = [(v['noun'], v['gender']) for v in nouns]

# weed out the p gender (apparently a plural)
nouns = [n for n in nouns if n[1] in {'f', 'm', 'n'}]

# example data
print('Total nouns:', len(nouns))
nouns[100:105]
import numpy as np

def rule_applies(rule, word):
    """Checks whether a rule given as a dict with keys f, m, n containing 
    values as arrays of ending applies to the word."""
    
    for endings in rule.values():
        for ending in endings:
            if word.endswith(ending):
                return True
    
    return False

def rule_predict(rule, word):
    # estimates ending for a given word
    result = 'f'  # default: feminine gender
    fitting_ending_length = 0  # longer ending takes precedence over shorter one
    
    for gender, endings in rule.items():
        for ending in endings:
            if word.endswith(ending) and len(ending) > fitting_ending_length:
                result = gender
                fitting_ending_length = len(ending)
    
    return result

def rule_confusion(rule):
    """Finds all endings that could be confused in the rule"""
    endings = [(e, g) for g in rule for e in rule[g]]
    
    all_confusion = []
    for end, gen in endings:
        confusion = []  # which endings can be confused?
        for end2, gen2 in endings:
            if end2.endswith(end) and len(end2) > len(end) and gen2 != gen:
                confusion.append(end2 + "(" + gen2 + ")")
        
        if confusion:
            all_confusion.append(end + "(" + gen + ") can be confused with " + ', '.join(confusion))
    
    return all_confusion

def rule_quality(rule, all_nouns, per_ending_accuracy=False, rule_analysis=False):
    """For a rule given as a dict with keys f, m, n containing 
    values as arrays of ending, evaluate this rule on a set of 
    nouns given as a list of tuples (noun str, gender \in m,f,n)
    
    Returns a dict with various rule performance metrics, such as:
    - Coverage: fraction of all supplied nouns, for which the rule
        applies;
    - Accuracy: overall accuracy of the rule on nouns where the
        rule applies. 
    """
    
    # result will be a dict with various metrics
    result = {}
    
    # first, select all the nouns that rule can be applied to
    nouns = [n for n in all_nouns if rule_applies(rule, n[0])]
    
    # percentage of all nouns covered by the rule
    result['Coverage %'] = round(len(nouns) / len(all_nouns), 3)
    result['Coverage count'] = len(nouns)
    result['Total count'] = len(all_nouns)
    
    # make estimations with rule
    y = np.array([gender for word, gender in nouns])  # ground truth
    y_pred = np.array([rule_predict(rule, word) for word, _ in nouns])  # estimations with rule
    # accuracy of the rule
    result['Accuracy'] = round(np.mean(y == y_pred), 3)
    
    # get the accuracy per class
    per_class_stats = {}
    for clas in set(y):
        I = y == clas
        per_class_stats[clas] = {
            'accuracy': round(np.mean(y[I] == y_pred[I]), 3),
            'instances': np.sum(I)
        }
    result['Gender accuracy'] = per_class_stats
    
    # get accuracy per ending of noun
    per_ending_stats = {}
    for gender, endings in rule.items():
        for ending in endings:
            I = np.array([word.endswith(ending) for word, gender in nouns])
            per_ending_stats[ending] = {
                'accuracy': round(np.mean(y[I] == y_pred[I]), 3),
                'gender': gender,
                'instances': np.sum(I)
            }
    
    result['Ending count'] = len(per_ending_stats)
    
    if per_ending_accuracy:
        result['Ending accuracy'] = per_ending_stats
    
    return result
from pprint import pprint
result = rule_quality(baseline_rule, nouns, True)
pprint(result, width=120)
def derive_rule(nouns, min_accuracy=0.8, min_nouns=50):
    # this will contain bins for every ending
    ending_bins = {}

    # make counts for all endings
    for word, gender in nouns:
        word = word.lower()  # ending might include first letter of word
        for ending_len in [1, 2, 3, 4, 5, 6]:
            ending = word[-ending_len:]
            if ending not in ending_bins:
                ending_bins[ending] = {'f':0, 'm':0, 'n':0}
            ending_bins[ending][gender] += 1
    
    # calculate statistics for every ending
    endings = []
    for ending, count in ending_bins.items(): 
        endings.append({
            "ending": ending,
            "nouns": sum(count.values()),
            "accuracy": max(count.values()) / sum(count.values()),
            "gender": max(count.keys(), key=count.get)
        })
    
    # filter out the endings
    endings = [
        e for e in endings if e['nouns'] >= min_nouns and e['accuracy'] >= min_accuracy
    ]
    
    # make a rule out of this
    rule = {
        gender: [stats['ending'] for stats in endings if stats['gender'] == gender]
        for gender in ['f', 'm', 'n']
    }
    
    # remove endings within one gender that contain other ending
    for gender, endings in rule.items():
        rule[gender] = [
            e for e in endings 
            if not any([  # look for any ending which is shorter and current one's ending
                True for e2 in endings if len(e2) < len(e) and e.endswith(e2)
            ])
        ]
    
    return rule

pprint(derive_rule(nouns, min_accuracy=0.8, min_nouns=50))
rule = derive_rule(nouns, min_accuracy=0.8, min_nouns=50)
result = rule_quality(rule, nouns, per_ending_accuracy=False)
pprint(result, width=120)
def experiment(min_accuracy=0.8, min_nouns=50):
    # derive the rule itself
    rule = derive_rule(nouns, min_accuracy, min_nouns)
    
    print('Quality or rule:')
    pprint(rule_quality(rule, nouns, per_ending_accuracy=False), width=120)
    
    # print the rule
    print('Derived rule:')
    pprint(rule, compact=True)
    
    print('Any confusing endings:')
    pprint(rule_confusion(rule))
experiment(0.71, 37)
experiment(0.66, 27)
experiment(0.64, 16)