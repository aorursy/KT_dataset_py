# Our dependencies

from graphviz import Digraph

import numpy as np

import pandas as pd

import ipywidgets as widgets

from ipywidgets import interactive

from IPython.display import display

import asyncio
import asyncio

from ipykernel.eventloops import register_integration



@register_integration('asyncio')

def loop_asyncio(kernel):

    '''Start a kernel with asyncio event loop support.'''

    loop = asyncio.get_event_loop()



    def kernel_handler():

        loop.call_soon(kernel.do_one_iteration)

        loop.call_later(kernel._poll_interval, kernel_handler)



    loop.call_soon(kernel_handler)

    try:

        if not loop.is_running():

            loop.run_forever()

    finally:

        loop.run_until_complete(loop.shutdown_asyncgens())

        loop.close()

%gui asyncio
#Some special chars

lda = u"\u03BB"

cdot = u"\u00B7"
class DFA:

    def __init__(self, alphabet, states_name, initial_state, final_states, transitions):

        self.transitions = transitions.astype(int)

        self.states_name = states_name

        self.N = len(states_name)

        self.alphabet = alphabet

        self.rev_alphabet = {s: i for i, s in enumerate(self.alphabet)}

        self.initial_state = initial_state

        self.final_states = final_states

        self.graph = self.make_dotgraph()

        

    def make_dotgraph(self):

        graph = Digraph(comment='The Round Table', graph_attr={"rankdir": "LR"})

        graph.attr('node', shape='circle')

        graph.node(".", shape="point")

        

        for i_s, s in enumerate(self.states_name):

            graph.node(str(i_s), s, shape="doublecircle" if i_s in self.final_states else None)

        

        graph.edge('.', str(self.initial_state))

        

        for i_state in range(self.N):

            to_state = {}

            for i_transition, transition in enumerate(self.alphabet):

                i_next_state = self.transitions[i_state][i_transition]

                to_state[i_next_state] = to_state.get(i_next_state, []) + [transition]

            for i_next_state, transitions in to_state.items():

                graph.edge(str(i_state), str(i_next_state), label=",".join(transitions))

        

        return graph

    

    def accept_word(self, word):

        state = self.initial_state

        for letter in word:

            state = self.transitions[state][self.rev_alphabet[letter]]

        return state in self.final_states
class LSTAR:

    SAS = "S.A\S"

    S = "S"

        

    def __init__(self, alphabet):

        self.alphabet = alphabet

        self.table = pd.DataFrame([[None, LSTAR.S]], index=[[lda], [lda]], columns=["result", "group"])

        self.table.index.set_names(["word", "experiment"], inplace=True)

        self.table.sort_index(level=["word", "experiment"], ascending=[1, 1], sort_remaining=True, inplace=True)

        

        self.all_results = {}

        

    def get_result(self, word, experiment):

        # Defaults to None if word has not been calculated yet

        return self.all_results.get(LSTAR.concat(word, experiment), None)

    

    def get_experiments(self):

        return self.table.index.get_level_values("experiment").unique()

    

    def concat(prefix, suffix):

        res = (prefix + suffix).replace(lda, '')

        return res if len(res) > 0 else lda

        

    def to_angluin_form(self):

        return self.table.set_index("group", append=True).unstack(level="experiment").reorder_levels([1, 0]).sort_index()

        

    def compute_successors(self):

        experiments = self.get_experiments()

        

        # Get the words of the first group of rows in the table

        S_rows = self.table[self.table['group'] == LSTAR.S]

        S_words = S_rows.index.get_level_values("word").unique()

        

        # This is the S·A step

        successors = set([LSTAR.concat(word, letter) for word in S_words for letter in self.alphabet])

        

        # We want to compute S·A\S, so we have to remove S from S·A

        successors = [succ for succ in successors if succ not in S_words]

                

        # errors='ignore' to not stop if we have not row in S·A\S yet

        new_table = self.table.drop(self.table[self.table['group'] == LSTAR.SAS].index, errors='ignore')

        

        for word in successors:

            for experiment in experiments:

                new_table.loc[(word, experiment), :] = [self.get_result(word, experiment), LSTAR.SAS]

        

        return new_table

    

    def get_holes(self):

        # put the experiences in the index to get only one column, with either a value or a "hole"

        missing_rows = self.table[self.table['result'].isnull()]

        

        # get the missing values as a list

        holes = list(missing_rows.index.values)

        

        return holes

    

    async def get_and_fills_holes(self):

        # get the table missing values

        holes = self.get_holes()

        

        # ask the user to fill those missing values

        concat_holes = [LSTAR.concat(word, experience) for word, experience in holes]

        answers = await LSTAR.prompt_words(concat_holes)

        holes_and_answers = list(zip(holes, answers))

        

        # put the user answers to the word+experience (concatenated) requests in our results list

        # and fill the table with the answers

        self.all_results.update(dict(zip(concat_holes, answers)))

        

        for (word, experience), answer in holes_and_answers:

            self.table.loc[(word, experience), 'result'] = answer

        



    def make_concat_results(self):

        # Let's get the distinct answers of the words in the potential states section "S"

        # remember that each state is labelled by a word that gets to this state

        # so here 'states' is actutally a list of words

        concat_results = self.table.copy()

        concat_column = self.table['result'].unstack(level="experiment").apply(

            lambda row: "".join(map(str, map(int, row))), axis=1

        )

        concat_results.index = concat_results.index.droplevel(level="experiment")

        concat_results = concat_results[~concat_results.index.duplicated(keep='first')]

        concat_results['concat'] = concat_column

        return concat_results

            

    def to_dfa(self):

        concat_results = self.make_concat_results()

        

        df_states = concat_results[concat_results['group'] == LSTAR.S][['concat']].drop_duplicates()

        states = df_states.index.get_level_values("word").tolist()

        

        transitions = np.ndarray((len(states), len(self.alphabet)))

        for i_state, state in enumerate(states):

            for i_letter, letter in enumerate(self.alphabet):

                

                # get the concatenated results of the word after transition

                # this will help us get the state corresponding to these kind

                # of results

                word_after_transition = LSTAR.concat(state, letter)

                row_value = concat_results.loc[word_after_transition, 'concat']

                                

                next_state = df_states[(df_states['concat'] == row_value)].index.get_level_values("word")[0]

                

                i_next_state = states.index(next_state)

                transitions[i_state][i_letter] = i_next_state

                

        # get the first state, ie the one accessed by the empty word 

        i_initial_state = states.index(lda)

        

        # get the final states, ie the one whose experiment lambda (empty word) directly lead to an accepting state

        true_lambda_result = self.table.query("experiment == '"+lda+"' and group == '"+LSTAR.S+"' and result == True")

        i_final_state_words = [states.index(word) for word in true_lambda_result.index.get_level_values("word") if word in states]

        

        return DFA(self.alphabet, states, i_initial_state, i_final_state_words, transitions)

    

    

    def close_table(self):

        

        # makes another column to held the full experiment results in a cell, easier to compare rows (=words)

        concat_df = self.make_concat_results()

        

        # what is in the S group

        results_in_s = concat_df[concat_df['group'] == 'S'][['concat']].drop_duplicates()

        

        # what is in S and SA (successors) ?

        results_in_both = concat_df[['concat']].drop_duplicates()

        

        # what is in the successors and not in the potential states

        

        is_not_in_s = (pd.merge(results_in_s, results_in_both, how="outer", indicator=True, left_index=True, right_index=True)['_merge'] == 'right_only')

        df_misplaced = results_in_both[is_not_in_s.reindex(results_in_both.index)]

        misplaced_words = df_misplaced.index.get_level_values('word').tolist()

        

        # for those words going outside of the potential states, we add them to the list possible state-words

        # (ie states that are accessibles directly by executing the words that led to their creations)

        for word in misplaced_words:

            self.table.loc[word, 'group'] = LSTAR.S

            

        # return if there was no such misplace word, and so if the table was already closed

        return len(misplaced_words) > 0

    

    def make_consistent_table(self):

        experiments = self.get_experiments()

        

        concat_df = self.make_concat_results()

        

        df_S = concat_df[concat_df['group'] == LSTAR.S]

                

        for concat in df_S['concat'].unique():

            same_state_words = df_S[df_S['concat'] == concat].index.values.tolist()

            pairs = [(w1, w2) for i, w1 in enumerate(same_state_words) for w2 in same_state_words[i+1:]]



            for w1, w2 in pairs:                

                for letter in self.alphabet:

                    next_w1 = LSTAR.concat(w1, letter)

                    next_w2 = LSTAR.concat(w2, letter)

                    

                    # If there is an inconsistency, ie two words leading to the same state would either quit, 

                    # or stay in this state when concatenated with one more letter, although they should

                    # respond the same way to the transition

                    rows_equal = (self.table.loc[next_w1]['result'] == self.table.loc[next_w2]['result'])

                    if not rows_equal.all():

                        for inconsistent_experiment in (rows_equal[~rows_equal]).index.values:

                            new_experiment = LSTAR.concat(letter, inconsistent_experiment)

                            # print("Possible new experiment {}: {}·{}".format(new_experiment, inconsistent_experiment, letter))

                            

                            # We dont add an experiment that is already in the list

                            if new_experiment in experiments:

                                print("New experiment {} already in the list".format(new_experiment))

                                continue

                                

                            # print("Inconsistency found: {} and {} go to different states through {}·{}".format(w1, w2, inconsistent_experiment, letter))

                            for word in self.table.index.get_level_values("word"):

                                self.table.loc[(word, new_experiment), :] = [self.get_result(word, new_experiment), self.table.loc[word].iloc[0]['group']]

                            return True

                

        return False

        

    def add_word_to_test(self, word):

        experiments = self.get_experiments()

        prefixes = [word[:i+1] for i in range(len(word))]

        for p in prefixes:

            for experiment in experiments:

                self.table.loc[(p, experiment), :] = [self.get_result(p, experiment), LSTAR.S]

    

    async def prompt_words(words):

        # Here we define our interactive widgets

        button_y = widgets.Button(description="Yes")

        button_y.on_click(lambda arg: click_answer.set_result(True))

        button_n = widgets.Button(description="No")

        button_n.on_click(lambda arg: click_answer.set_result(False))

        label = widgets.Label("", layout=widgets.Layout(width='50%'))

        layout = widgets.VBox([label, widgets.HBox([button_y, button_n])])

        

        # And display them

        display(layout)

        

        results = []

        

        for word in words:

            # Future that is going to be completed upon button click

            click_answer = asyncio.Future()

            label.value = "Is '{}' in the langage ?".format(word)

            results.append(await click_answer)

        

        layout.close()

        return results

    

    async def prompt_dfa(dfa):

        answer = asyncio.Future()

        

        button_y = widgets.Button(description="This is the graph !")

        button_y.on_click(lambda arg: answer.set_result(True))

        text = widgets.Text(description="Word")

        text.on_submit(lambda sender: answer.set_result(text.value))

        label = widgets.Label("Is it this graph ? \nIf not, enter a word that is not accepted and should, or is accepted and shouldn't",

                             layout=widgets.Layout(width='100%'))

        

        html_graph = widgets.HTML(dfa.graph._repr_svg_())

        layout = widgets.VBox([label, html_graph, widgets.HBox([button_y, text])])

        display(layout)

        

        res = await answer

        

        layout.close()

        

        return res

    

    async def _run(self):

        self.table = self.compute_successors()

        await self.get_and_fills_holes()

        

        

        while True:



            ready_for_equivalence_request = False

            while not ready_for_equivalence_request:



                was_not_closed = self.close_table()



                # The conditions could be made shorter but a few more variables and lines

                # are added for the sake for readability

                to_fill = False



                if was_not_closed:

                    to_fill = True

                else:

                    was_not_consistent = self.make_consistent_table()



                    if was_not_consistent:

                            to_fill = True

                if to_fill:

                    self.table = self.compute_successors()

                    await self.get_and_fills_holes()

                else:

                    ready_for_equivalence_request = True

            

            dfa = self.to_dfa()

            answer = await LSTAR.prompt_dfa(dfa)

            

            if answer is True:

                print("🎊 Yeay! 🎊")

                display(dfa.graph)

                break

            else:

                self.add_word_to_test(answer)

                self.table = self.compute_successors()

                await self.get_and_fills_holes()

    

    def run(self):

        asyncio.ensure_future(self._run())
model = LSTAR(alphabet=['a', 'b'])
model.run()