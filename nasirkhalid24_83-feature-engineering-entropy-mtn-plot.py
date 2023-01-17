!conda install -y -c bioconda viennarna
data = """#!/usr/bin/perl -w

# -*-Perl-*-

# Last changed Time-stamp: <2005-03-09 19:38:25 ivo> 

# produce Pauline's mountain representation from bracket notation

# use e.g. as  RNAfold < foo.seq | b2mt | xmgr -pipe

# definition: h=number of base pairs enclosing base

use strict;

while (<>) {

    print if (s/>/#/);



    next unless (/\.\.\./);

    next if (/\[/);   # don't process output from partition function

    chop;

    my @F=split(//,$_);

    my $p=0; my $h=0;

    foreach my $i (@F) {

    $h-- if ($i eq ')');

    $p++;

    printf("%4d %4d\n",$p,$h);

    $h++ if ($i eq '(');# increase $h *after* printing

    }

    print "&\n";

}



=head1 NAME



b2mt - produce coordinates for a mountain plot from bracket notation



=head1 SYNOPSIS



  b2mt.pl < seq.fold > mountain.dat



=head1 DESCRIPTION



read a secondary structures in bracket notation as output by RNAfold,

and compute coordinates for a mountain plot as introduced by Pauline Hogeweg.

The output is suitable for graphing  with xmgrace, e.g.:

C< RNAfold < foo.seq | b2mt.pl | xmgrace -pipe>



=head1 AUTHOR



Ivo L. Hofacker <ivo@tbi.univie.ac.at>"""



with open('b2mt.pl', 'w') as f:

      f.write(data)
data = """#!/usr/bin/perl -w

# -*-Perl-*-

# Last changed Time-stamp: <2008-08-26 16:04:00 ivo>

# produce Pauline Hogeweg's mountain representation *_dp.ps files

# writes 3 sets of x y data separated by a "&"

# the first two sets are mountain representations from base pair probabilities

# and mfe structure, respectively.

# definition: mm[i],mp[i] = (mean) number of base pairs enclosing base i

# third set a measure of well-definedness: the entropy of the pair probs of

# base i, sp[i] = -Sum p_i * ln(p_i). Well-defined regions have low entropy.

#

# use e.g. as  mountain.pl dot.ps | xmgrace -pipe



use strict;

our (@mm, @mp, @pp, @sp, @p0, $i, $max, $length, $do_png);  # perl5 only



my $sep = "&";   # xmgr uses & to separate data sets



if (@ARGV && ($ARGV[0] eq '-png')) {

    eval "use Chart::Lines";

    die($@,

    "\nCould not load the Chart::Lines module required with -png option\n")

    if $@;

    $do_png=1;

    shift;

}





while (<>) {

    chomp;

    if (/\/sequence \{ \((\S*)[\\\)]/) {

    my $seq = $1;           # empty for new version

    while (!/\) \} def/) {  # read until end of definition

        $_ = <>;

        /(\S*)[\\\)]/;      # ends either with `)' or `\'

       $seq .= $1;

    }

    $length = length($seq);

    next;

    }



    next unless /(\d+) (\d+) (\d+\.\d+) (.box)$/;

    my ($i, $j, $p, $id) = ($1,$2,$3,$4);

    if ($id eq "ubox") {

    $p *= $p;           # square it to probability

    $mp[$i+1] += $p;

    $mp[$j]   -= $p;

    my $ss = $p>0 ? $p*log($p) : 0;

    $sp[$i] += $ss;

    $sp[$j] += $ss;

    $pp[$i] += $p;

    $pp[$j] += $p;

    }

    if ($id eq "lbox") {

    $mm[$i+1]++;

    $mm[$j]--;

    }

}

$mp[0] = $mm[0] = $max = 0;

for ($i=1; $i<=$length; $i++) {

    no warnings;

    $mp[$i]+=$mp[$i-1];

    $max = $mp[$i] if ($mp[$i]>$max);

    $mm[$i]+=$mm[$i-1];

    $max = $mp[$i] if ($mp[$i]>$max);

    $sp[$i] += (1-$pp[$i])*log(1-$pp[$i]);

}



if ($do_png) {

    my $width =  800;

    my $height = 600;



    # FIXME: legend_lables when doing mfe only

    my $skip = 10**(int (log($length)/log(10.) - 0.5));

    my $obj = Chart::Lines->new( $width, $height );

    $obj->set ('title' => $ARGV,

           'x_label' => 'Position',

           'y_label' => 'Height',

           'min_val' => 0,

           'precision' => 0,

           'legend_labels' => ['mfe', 'pf'],

           'skip_x_ticks' => $skip);



    $obj->add_dataset ((0..$length));



    $obj->add_dataset (@mp);

    $obj->add_dataset (@mm);

    $obj->png("mountain.png");



} else {

    # print the results for plotting

    for ($i=1; $i<=$length; $i++) {

    printf("%4d  %7.5g\n", $i, $mp[$i]);

    }

    print "$sep\n";



    for ($i=1; $i<=$length; $i++) {

    printf("%4d  %4d\n", $i, $mm[$i]);

    }

    print "&\n";

    my $log2 = log(2);

    for ($i=1; $i<=$length; $i++) {

    printf("%4d  %7.5g\n", $i, -$sp[$i]/$log2);

    }

}



=head1 NAME



    mountain - produce coordinates for a mountain plot from a dot plot



=head1 SYNOPSIS



    mountain.pl myseq_dp.ps > mountain.dat



=head1 DESCRIPTION



    reads pair proabilities and MFE structure from a probability dot

    plot as produced by C<RNAfold -p>, and produces x-y data suitable

    for producing a mountain plot using standard xy-plotting programs.



    Output consists of 3 data sets separated by a line containing only

    the C<&> character. The first two sets are mountain representations

    computed from base pair probabilities and mfe structure, respectively.

    For the mfe case the moutain plot graphs the number base pairs

    enclosing a position k, in case of pair probabilities we use the average

    number of base pairs computed as m_k = \Sum_i<k<j p_ij.

    The third set contains the positional entropy, which provides a measure

    of local structural welldefinedness, s_i = -\Sum_j p_ij * ln(p_ij).



    The output is suitable for graphing with xmgrace, e.g.:

    C< RNAfold -p < foo.seq; mountain.pl foo_dp.ps | xmgrace -pipe>



=head1 AUTHOR



Ivo L. Hofacker <ivo@tbi.univie.ac.at>"""



with open('mountain.pl', 'w') as f:

      f.write(data)
!chmod 755 ./b2mt.pl

!chmod 755 ./mountain.pl
import pandas as pd

import numpy as np



import tempfile

import os 

import re

import subprocess

import tempfile



import matplotlib.pyplot as plt
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

sequence = train.iloc[0]['sequence']
sequence
def RNAstructures(sequence, length):

    tf = tempfile.NamedTemporaryFile(delete=False)



    tf.name += '.seq'

    with open(tf.name, 'w') as f:

        f.write(sequence)



    cmd = 'RNAfold -p -d2 --noLP < ' + tf.name

    cmd_result = subprocess.check_output(cmd, shell=True).decode().split('\n')



    # Remove intermediate files

    subprocess.check_output('rm -r ./rna.ps', shell=True)

    subprocess.check_output('rm -r ./dot.ps', shell=True)

    

    # Return only the MFE and Centroid structures

    return [x[:length] for x in cmd_result if '(' in x[:length] and ')' in x[:length] and '{' not in x[:length] and ',' not in x[:length]]



# some times structures might be same best to remove any duplicates

structures = RNAstructures(sequence, 107)

structures
def RNAeval(sequence, structure):

    tf = tempfile.NamedTemporaryFile(delete=False)

    ss = structure



    tf.name += '.fa'

    with open(tf.name, 'w') as f:

        f.write('>' + 'rna_seq' + '\n')

        f.write(sequence + '\n')

        f.write(ss + '\n')



    dopt = ' -v -d2 '

    if False:

        dopt = ' -d0 '



    cmd = 'RNAeval ' + dopt + ' < ' + tf.name

    

    # Returns free energy values for each loop

    return subprocess.check_output(cmd, shell=True).decode().split('\n')



RNAeval(sequence, structures[1])
def result_decoder(res):

    final_ = {}



    positives_ = []

    negatives_ = []



    for result in res[1:-3]:

        energy = int(result[-5:].strip())

        r = r"\((.*?)\)"



        if energy > 0:

            matches = re.findall(r, result)

            if len(matches) > 0:

                start, end = matches[0].split(",")

                positives_.append([int(start)-1, int(end)-1, energy])

            else:

                print(error_bby)

        else:

            matches = re.findall(r, result)

            if len(matches) > 0:

                start, end = matches[0].split(",")

                negatives_.append([int(start)-1, int(end)-1, energy])

            else:

                final_["external"] = energy



    final_["closed"] = positives_

    final_["outer"] = negatives_

    

    # Returns start, end and energy for all enclosed nucleotides in between start and end

    return final_



result_decoder(RNAeval(sequence, structures[1]))
def get_entropy(pls, entropy_arr):

    for points in pls["closed"]:

        val_ = points[-1] / (points[1]-points[0]-1)

        if val_ == -170:

            print(points)

        entropy_arr[points[0]+1:points[1]] += val_



    for points in pls["outer"]:

        val_ = points[-1] / 2

        entropy_arr[points[0]+1] += val_

        entropy_arr[points[1]-1] += val_



    remaining_ = np.where(entropy_arr == 0)[0]

    val_ = pls["external"] / len(remaining_)

    entropy_arr[np.where(entropy_arr == 0)[0]] = val_

    

    # Returns a vector of length of sequence with energy values are assigned to each nucleotide

    # such that the sum of entire vector is equal to Free Energy of the structure 

    return entropy_arr



fig, axs = plt.subplots(2, 1)



free_energy_0 = get_entropy(result_decoder(RNAeval(sequence, structures[0])), np.zeros(len(structures[0])))

free_energy_1 = get_entropy(result_decoder(RNAeval(sequence, structures[1])), np.zeros(len(structures[1])))



axs[0].plot(free_energy_0)

axs[0].set_title('Free Energy of Structure 1\n'+str(structures[0])+'\nTotal of Structure = '+str(free_energy_0.sum()))



axs[1].plot(free_energy_1)

axs[1].set_title('Free Energy of Structure 2\n'+str(structures[1])+'\nTotal of Structure = '+str(free_energy_1.sum()))



fig.tight_layout()

plt.show()
def RNAmountain(sequence, structure):

    tf_m = tempfile.NamedTemporaryFile(delete=False)

  

    tf_m.name += '.fa'

    with open(tf_m.name, 'w') as f:

        f.write(structure)

        

    cmd_m = './b2mt.pl ' + tf_m.name

    

    result = subprocess.check_output(cmd_m, shell=True).decode().split('\n')

    

    graph = []

    for val in result:

        num_ = val.strip().split(' ')[-1]

        if num_ == '&':

            graph.append(0)

        elif num_ != '':

            graph.append(float(num_))



    return np.array(graph)



fig, axs = plt.subplots(2, 1)



mountain_plot_0 = RNAmountain(sequence, structures[0])



axs[0].plot(mountain_plot_0)

axs[0].set_title('Mountain Plot of Structure 1\n'+str(structures[0]))



mountain_plot_1 = RNAmountain(sequence, structures[1])



axs[1].plot(mountain_plot_1)

axs[1].set_title('Mountain Plot of Structure 2\n'+str(structures[1]))



fig.tight_layout()

plt.show()
def RNAenergies(sequence, structure):

    tf_te = tempfile.NamedTemporaryFile(delete=False)



    tf_te.name += '.seq'

    with open(tf_te.name, 'w') as f:

        f.write(sequence + '\n')



    cmd_te = 'RNAfold -p < ' + tf_te.name



    subprocess.check_output(cmd_te, shell=True).decode() #dot.ps



    result = subprocess.check_output('./mountain.pl dot.ps', shell=True).decode().split('\n')

    

    subprocess.check_output('rm -r ./dot.ps', shell=True)

    subprocess.check_output('rm -r ./rna.ps', shell=True)



    all_graphs = []

    graph = []



    for val in result:

        num_ = val.strip().split(' ')[-1]

        if num_ == '&':

            all_graphs.append(np.array(graph))

            graph = []

        elif num_ != '':

            graph.append(float(num_))



    all_graphs.append(np.array(graph))

    graph = []



    # Remove extra mountain plot

    return np.delete(np.array(all_graphs), (1), axis=0)



# Same for both

energies = RNAenergies(sequence, structures[0])



fig, axs = plt.subplots(2, 1)



axs[0].plot(energies[0])

axs[0].set_title('Thermodynamic Ensemble of Sequence\n'+str(structures[0]))



axs[1].plot(energies[1])

axs[1].set_title('Positional Entropy\n'+str(structures[1]))



fig.tight_layout()

plt.show()