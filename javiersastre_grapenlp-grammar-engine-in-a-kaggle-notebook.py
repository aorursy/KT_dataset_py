# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!dpkg -i ../input/libgrapenlp/libgrapenlp_2.8.0-0ubuntu1_xenial_amd64.deb
!dpkg -i ../input/libgrapenlp/libgrapenlp-dev_2.8.0-0ubuntu1_xenial_amd64.deb
!pip install pygrapenlp
import json
from collections import OrderedDict
from pygrapenlp import u_out_bound_trie_string_to_string
from pygrapenlp.grammar_engine import GrammarEngine
base_dir = os.path.join('..', 'input', 'grammar')
grammar_pathname = os.path.join(base_dir, 'test_grammar.fst2')
bin_delaf_pathname = os.path.join(base_dir, 'test_delaf.bin')
grammar_engine = GrammarEngine(grammar_pathname, bin_delaf_pathname)

def native_results_to_python_dic(sentence, native_results):
    top_segments = OrderedDict()
    if not native_results.empty():
        top_native_result = native_results.get_elem_at(0)
        top_native_result_segments = top_native_result.ssa
        for i in range(0, top_native_result_segments.size()):
            native_segment = top_native_result_segments.get_elem_at(i)
            native_segment_label = native_segment.name
            segment_label = u_out_bound_trie_string_to_string(native_segment_label)
            segment = OrderedDict()
            segment['value'] = sentence[native_segment.begin:native_segment.end]
            segment['start'] = native_segment.begin
            segment['end'] = native_segment.end
            top_segments[segment_label] = segment
    return top_segments

sentence = 'this is a test sentence'
context = {}
native_results = grammar_engine.tag(sentence, context)
matches = native_results_to_python_dic(sentence, native_results)
print(json.dumps(matches, indent=4))
sentence = 'this is another test sentence'
context = {}
native_results = grammar_engine.tag(sentence, context)
matches = native_results_to_python_dic(sentence, native_results)
print(json.dumps(matches, indent=4))
sentence = 'this is a context test sentence'
context = {}
native_results = grammar_engine.tag(sentence, context)
matches = native_results_to_python_dic(sentence, native_results)
print(json.dumps(matches, indent=4))
sentence = 'this is a context test sentence'
context = {'context': 'true'}
native_results = grammar_engine.tag(sentence, context)
matches = native_results_to_python_dic(sentence, native_results)
print(json.dumps(matches, indent=4))